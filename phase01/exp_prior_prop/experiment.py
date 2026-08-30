"""experiment.py — PRIOR-PROP training + eval (ТЗ П1-П5).

Usage: python experiment.py DP|DP-noprop|DP-rand|C-cap|PM|SP [steps]
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
for p in (HERE, PHASE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import build_model, count_params, VOCAB, W, D
from exp_memory_selector.experiment import load_chars, build_tokenizer, MAX_TRAIN

# ----------------------------------------------------------------- config
BATCH = 64
STEPS = 8000
LR = 5e-4
WARMUP = 1000
N_VAL = 2000
N_EVAL = 5000
EVAL_BATCH = 200
GAP_TRIGGER = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------- data
print("BPE...", flush=True)
tok = build_tokenizer()
V = tok.get_vocab_size()
train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}", flush=True)


# ----------------------------------------------------------------- train
def train(model, name):
    params = sum(p.numel() for p in model.parameters())
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        p = (step - WARMUP) / max(1, STEPS - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    rng = np.random.default_rng(1)
    maxstart_v = len(train_ids) - W - 1
    val_starts = np.sort(rng.choice(maxstart_v, size=N_VAL, replace=False))
    t0 = time.time()
    gap_log = []
    stopped = False
    for s in range(STEPS):
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i+W] for i in bi]), dtype=torch.long, device=DEVICE)
        Y = torch.tensor([train_ids[i+W] for i in bi], dtype=torch.long, device=DEVICE)
        opt.zero_grad()
        logits = model(X)
        loss = lossf(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 2000 == 0:
            tr_loss = loss.item()
            model.eval()
            with torch.no_grad():
                vnll = 0.0
                for s0 in range(0, N_VAL, EVAL_BATCH):
                    e = min(s0 + EVAL_BATCH, N_VAL)
                    Xv = np.zeros((e - s0, W), dtype=np.int64)
                    Yv = np.zeros(e - s0, dtype=np.int64)
                    for kk, i in enumerate(val_starts[s0:e]):
                        Xv[kk] = train_ids[i:i+W]; Yv[kk] = train_ids[i+W]
                    lo = model(torch.tensor(Xv, dtype=torch.long, device=DEVICE))
                    vnll += float(lossf(lo, torch.tensor(Yv, dtype=torch.long, device=DEVICE)).item()) * (e - s0)
                val_loss = vnll / N_VAL
            model.train()
            gap = tr_loss - val_loss
            gap_log.append({"step": s, "train": tr_loss, "val": val_loss, "gap": gap})
            print(f"[{name}] [{s}] tr={tr_loss:.3f} val={val_loss:.3f} gap={gap:+.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
            # П4: gap > trigger -> stop
            if gap > GAP_TRIGGER and s > WARMUP:
                print(f"[{name}] GAP TRIGGER ({gap:+.3f} > {GAP_TRIGGER}) — early stop", flush=True)
                stopped = True
                break
    return params, gap_log, stopped


# ----------------------------------------------------------------- eval
def eval_model(model, name):
    model.eval()
    rng = np.random.default_rng(42)
    maxstart = len(test_ids) - W - 1
    te_starts = np.sort(rng.choice(maxstart, size=N_EVAL, replace=False))
    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=np.int64)
    with torch.no_grad():
        for s0 in range(0, N_EVAL, EVAL_BATCH):
            e = min(s0 + EVAL_BATCH, N_EVAL)
            X = np.zeros((e - s0, W), dtype=np.int64)
            for k, i in enumerate(te_starts[s0:e]):
                X[k] = test_ids[i:i+W]; y_te[s0+k] = test_ids[i+W]
            logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long, device=DEVICE)).cpu().numpy()
    ppl = float(np.exp(np.mean([-logits_te[k, y_te[k]] for k in range(N_EVAL)])))
    return ppl, logits_te, y_te


def retrieval_accuracy(model, distances=(16, 64, 256, 1024), n_trials=200, noise_frac=0.9):
    """KEY at distance L -> predict KEY (ТЗ: дальние связи)."""
    model.eval()
    rng = np.random.default_rng(7)
    key_candidates = list(range(150, V))
    counts = {}
    # find keys present in train
    from collections import Counter
    tr_counts = Counter(train_ids)
    cands = [t for t in key_candidates if tr_counts.get(t, 0) >= 20]
    if not cands:
        return {}
    res = {}
    for L in distances:
        acc = 0.0
        n_tested = 0
        for _ in range(n_trials):
            key = int(rng.choice(cands))
            # window of W; KEY at distance L BEFORE the prediction point.
            # if L >= W the key falls OUTSIDE the window — honest test
            # (model cannot see it: accuracy should be ~ random).
            i = int(rng.integers(W + L, len(test_ids) - 1))
            ctx = list(test_ids[i - W:i])
            key_pos_in_win = W - L          # position of key inside window
            if key_pos_in_win >= 0:
                ctx[key_pos_in_win] = key   # key visible at distance L
            # else: key outside window — nothing to place
            # optionally inject noise
            n_noise = int(W * noise_frac)
            noise_idx = rng.choice(W, size=n_noise, replace=False)
            for ni in noise_idx:
                if ni == key_pos_in_win:
                    continue
                ctx[ni] = int(rng.integers(1, 100))
            X = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                logits = model(X).cpu().numpy()[0]
            pred = int(np.argmax(logits))
            n_tested += 1
            if pred == key:
                acc += 1.0
        res[L] = acc / max(1, n_tested)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["DP", "DP-noprop", "DP-rand", "C-cap", "PM", "SP"])
    ap.add_argument("steps", nargs="?", type=int, default=STEPS)
    args = ap.parse_args()
    if args.steps != STEPS:
        STEPS = args.steps

    model = build_model(args.config, vocab=V)
    params, gap_log, stopped = train(model, args.config)
    # save checkpoint for analyze_links.py
    torch.save(model.state_dict(), os.path.join(HERE, f"model_{args.config}.pt"))
    ppl, logits_te, y_te = eval_model(model, args.config)
    print(f"[{args.config}] params={params:,} PPL={ppl:.3f}", flush=True)

    # retrieval accuracy vs distance (главный тест)
    retr = retrieval_accuracy(model)
    print(f"[{args.config}] retrieval: " + ", ".join(f"L={L}:{a:.3f}" for L, a in retr.items()), flush=True)

    res = {"config": args.config, "params": params, "ppl": ppl,
           "retrieval": retr, "gap_log": gap_log, "stopped": stopped,
           "steps_done": gap_log[-1]["step"] if gap_log else 0}
    json.dump(res, open(os.path.join(HERE, f"results_{args.config}.json"), "w"), indent=2)
    print(f"[{args.config}] saved results_{args.config}.json", flush=True)
