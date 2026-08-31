"""experiment_pc.py — PC-микшер LM: диссипативная Пекоры-Кэрролл динамика.

Переиспользует данные, train-протокол, eval, order-3 гейт и честный
retrieval из exp_vq/experiment.py, но модель = PCLM (models_pc).
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.join(PHASE, "exp_memory_selector"))

from experiment import load_chars, make_bpe, MAX_TRAIN, VOCAB, W, D, BLOCKS as B
from models_pc import build_pc_model

STEPS = 6000
BATCH = 64
LR = 5e-4
WARMUP = 1000
N_EVAL = 5000


def build_order3(train_ids):
    prior = defaultdict(dict)
    for i in range(3, len(train_ids)):
        ctx = tuple(train_ids[i - 2:i])
        w = train_ids[i]
        d = prior[ctx]
        d[w] = d.get(w, 0) + 1
    return {k: dict(v) for k, v in prior.items()}


def gated_ppl(logits, targets, prior, V, ctx_tokens=None):
    N = len(logits)
    nll = np.zeros(N)
    for k in range(N):
        lp = logits[k]
        pm = np.exp(lp[targets[k]])
        if ctx_tokens is not None:
            ctx = ctx_tokens[k]
        else:
            ctx = tuple(targets[k - 2:k]) if k >= 2 else ()
        table = prior.get(ctx)
        if table:
            tot = sum(table.values())
            c = table.get(targets[k], 0)
            if c > 0:
                beta = tot / (tot + 1.0)
                nll[k] = -np.logaddexp(np.log1p(-beta) + np.log(pm),
                                        np.log(beta) + np.log(c / tot))
                continue
        nll[k] = -np.log(pm)
    return float(np.exp(np.mean(nll)))


def induction_retrieval(model, test_ids, distances=(16, 64, 128, 256), n_trials=200):
    """Честный индукционный тест + сбор распределения выбранных позиций драйвера."""
    model.eval()
    rng = np.random.default_rng(0)
    res = {}
    all_pos = []
    for L in distances:
        hits, miss = 0, 0
        pos_L = []
        for _ in range(n_trials):
            i = int(rng.integers(L + 2, len(test_ids) - L - 3))
            A = int(test_ids[i])
            B = int(test_ids[i + 1])
            # второй KEY на дистанции L
            j = i + L
            window = test_ids[j - W:j]
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model(X)
                if hasattr(model, 'last_driver_pos') and model.last_driver_pos is not None:
                    p = int(model.last_driver_pos[0].item())
                    pos_L.append(p)
                    all_pos.append(p)
            logits = out[0] if isinstance(out, tuple) else out
            pred = int(logits[0].argmax().item())
            if pred == B:
                hits += 1
            miss += 1
        res[L] = hits / max(1, miss)
    drv_stats = {}
    if all_pos:
        drv_stats["mean"] = float(np.mean(all_pos))
        drv_stats["tail_frac"] = float(np.mean([1 if p >= W - 32 else 0 for p in all_pos]))
    return res, drv_stats


def train(config, steps=STEPS, alpha=0.9, k_init=1.2, sync_steps=1, driver_mode="mean", temp=0.3):
    print(f"\n=== Training {config} (alpha={alpha}, k={k_init}, T={sync_steps}, drv={driver_mode}, temp={temp}) ===", flush=True)
    rng = np.random.default_rng(0)
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    train_ids = tok.encode(train_text).ids
    test_ids = tok.encode(test_text).ids
    print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}", flush=True)

    model = build_pc_model(config, V, alpha=alpha, k_init=k_init,
                           sync_steps=sync_steps, driver_mode=driver_mode,
                           temp=temp).to("cuda")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params={nparam:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n = len(train_ids) - W - 1
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()

    for step in range(1, steps + 1):
        if step < WARMUP:
            lr = LR * step / WARMUP
            for pg in opt.param_groups:
                pg['lr'] = lr

        model.train()
        s = rng.integers(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
        Y = torch.tensor([train_ids[i + W] for i in s], dtype=torch.long, device="cuda")

        logits = model(X)
        loss = lossf(logits, Y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 2000 == 0:
            dt = time.time() - t0
            print(f"  [{step}] loss={loss.item():.3f} ({dt:.0f}s)", flush=True)

    # ---- eval ----
    model.eval()
    # сохраняем чекпоинт для диагностики (скрытые состояния, селекция)
    torch.save(model.state_dict(), os.path.join(HERE, f"model_{config}.pt"))
    rng2 = np.random.default_rng(42)
    te_starts = np.sort(rng2.choice(len(test_ids) - W - 1, size=N_EVAL, replace=False))

    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=int)
    for k, s in enumerate(te_starts):
        X = torch.tensor([test_ids[s:s + W]], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model(X)
        logits_k = out[0] if isinstance(out, tuple) else out
        logits_te[k] = logits_k[0].float().cpu().numpy()
        y_te[k] = test_ids[s + W]

    lpm = torch.log_softmax(torch.tensor(logits_te), -1).numpy()
    mixer_ppl = float(np.exp(np.mean([-lpm[k, y_te[k]] for k in range(N_EVAL)])))

    prior = build_order3(train_ids)
    ctx_tokens = [tuple(test_ids[s + W - 2:s + W]) for s in te_starts]
    gated = gated_ppl(lpm, y_te, prior, V, ctx_tokens)

    # индукционный retrieval + распределение выбранных драйверов
    retrieval, drv_stats = induction_retrieval(model, test_ids)
    drv_mean = drv_stats.get("mean", -1.0)
    drv_tail_frac = drv_stats.get("tail_frac", -1.0)  # доля выбранных позиций в последних 32

    res = {
        "config": config, "params": nparam,
        "mixer_ppl": round(mixer_ppl, 3), "gated_ppl": round(gated, 3),
        "retrieval": {str(k): round(v, 4) for k, v in retrieval.items()},
        "drv_mean_pos": round(drv_mean, 2), "drv_tail_frac": round(drv_tail_frac, 3),
    }
    print(f"[{config}] mixer={mixer_ppl:.3f} gated={gated:.3f} "
          f"retrieval={retrieval} drv_mean={drv_mean:.1f} tail={drv_tail_frac:.3f}", flush=True)

    with open(os.path.join(HERE, f"results_{config}.json"), "w") as f:
        json.dump(res, f)
    print(f"[{config}] saved results_{config}.json", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["pc"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--k", type=float, default=1.2)
    ap.add_argument("--sync-steps", type=int, default=1)
    ap.add_argument("--driver", choices=["mean", "last", "top1", "soft", "crt", "sts_emb", "sts_h"], default="mean")
    ap.add_argument("--temp", type=float, default=0.3)
    args = ap.parse_args()
    train(args.config, args.steps, alpha=args.alpha, k_init=args.k,
          sync_steps=args.sync_steps, driver_mode=args.driver, temp=args.temp)