"""experiment.py — VQ-Phase: тренировка и оценка (ТЗ этап A).

Порядок запуска: python experiment.py baseline|vq_only|vq_kto|vq_aux|nochao
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
from models import build_model

STEPS = 6000
BATCH = 64
LR = 5e-4
WARMUP = 1000
N_EVAL = 5000


def build_order3(train_ids):
    """Order-3 prior table (context -> next counts)."""
    prior = defaultdict(dict)
    for i in range(3, len(train_ids)):
        ctx = tuple(train_ids[i - 2:i])
        w = train_ids[i]
        d = prior[ctx]
        d[w] = d.get(w, 0) + 1
    return {k: dict(v) for k, v in prior.items()}


def gated_ppl(logits, targets, prior, V, ctx_tokens=None):
    """Compute gated PPL with adaptive beta (exp09-style).
    ctx_tokens: список кортежей реальных токенов контекста (или используется targets)."""
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


def induction_retrieval(model, ids, distances=(16, 64, 128, 256), n_trials=200):
    """Честный индукционный retrieval (KEY->B, KEY на L)."""
    model.eval()
    rng = np.random.default_rng(0)
    res = {}
    for L in distances:
        hits, miss = 0, 0
        for _ in range(n_trials):
            pos = int(rng.integers(3, len(ids) - W - 2))
            A = ids[pos]
            B = ids[pos + 1]
            j = pos + L
            if j >= len(ids) - 1:
                continue
            if ids[j] != A:
                for d in range(-3, 4):
                    if pos + L + d < len(ids) - 1 and ids[pos + L + d] == A:
                        j = pos + L + d
                        break
            if ids[j] != A or j < W:
                continue
            X = torch.tensor([ids[j - W:j]], dtype=torch.long, device="cuda")
            with torch.no_grad():
                logits = model(X)
            if isinstance(logits, tuple):
                logits = logits[0]
            top = int(logits[0].argmax().item())
            if top == B:
                hits += 1
            miss += 1
        res[L] = hits / max(1, miss)
    return res


def train(config, steps=STEPS):
    print(f"\n=== Training {config} ===", flush=True)
    rng = np.random.default_rng(0)
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    train_ids = tok.encode(train_text).ids
    test_ids = tok.encode(test_text).ids
    print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}", flush=True)

    model = build_model(config, V).to("cuda")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params={nparam:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n = len(train_ids) - W - 1
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()

    for step in range(1, steps + 1):
        # Warmup + cosine LR
        if step < WARMUP:
            lr = LR * step / WARMUP
            for pg in opt.param_groups:
                pg['lr'] = lr

        model.train()
        s = rng.integers(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
        Y = torch.tensor([train_ids[i + W] for i in s], dtype=torch.long, device="cuda")

        out = model(X)
        if config == "baseline":
            logits = out
            loss = lossf(logits, Y)
        elif config == "vq_aux":
            logits, exact_logits, commit, cb, usage = out
            main_loss = lossf(logits, Y)
            aux_loss = lossf(exact_logits, Y)
            loss = main_loss + model.aux_w * aux_loss + model.vq.vq_w * (commit + cb)
        elif config == "nochao":
            logits, exact_logits, commit, cb, usage = out
            main_loss = lossf(logits, Y)
            aux_loss = lossf(exact_logits, Y)
            loss = main_loss + 0.1 * aux_loss + model.vq_w * (commit + cb)
        else:
            logits, commit, cb, usage, idx = out
            loss = lossf(logits, Y) + model.vq.vq_w * (commit + cb)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 2000 == 0:
            dt = time.time() - t0
            print(f"  [{step}] loss={loss.item():.3f} ({dt:.0f}s)", flush=True)

    # ---- eval ----
    model.eval()
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

    # mixer-only PPL
    lpm = torch.log_softmax(torch.tensor(logits_te), -1).numpy()
    mixer_ppl = float(np.exp(np.mean([-lpm[k, y_te[k]] for k in range(N_EVAL)])))

    # gated PPL (order-3 prior) — используем лог-вероятности, не сырые логиты
    prior = build_order3(train_ids)
    # передаём реальные токены контекста (последние 2 токена окна), а не targets
    ctx_tokens = [tuple(test_ids[s + W - 2:s + W]) for s in te_starts]
    gated = gated_ppl(lpm, y_te, prior, V, ctx_tokens)

    # induction retrieval
    retrieval = induction_retrieval(model, test_ids)

    # codebook stats
    cb_usage = 0.0
    if hasattr(model, 'vq') and hasattr(model.vq, 'codebook'):
        with torch.no_grad():
            dummy = torch.randint(0, V, (BATCH, W), device="cuda")
            e = model.embed_pos(dummy)
            _, idx, _, _, _ = model.vq(e)
            cb_usage = len(torch.unique(idx)) / model.vq.n_codes

    res = {
        "config": config, "params": nparam,
        "mixer_ppl": round(mixer_ppl, 3), "gated_ppl": round(gated, 3),
        "retrieval": {str(k): round(v, 4) for k, v in retrieval.items()},
        "codebook_usage": round(cb_usage, 3),
    }
    print(f"[{config}] mixer={mixer_ppl:.3f} gated={gated:.3f} "
          f"retrieval={retrieval} cb_usage={cb_usage:.3f}", flush=True)

    with open(os.path.join(HERE, f"results_{config}.json"), "w") as f:
        json.dump(res, f)
    print(f"[{config}] saved results_{config}.json", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["baseline", "vq_only", "vq_kto", "vq_aux", "nochao"])
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    train(args.config, args.steps)