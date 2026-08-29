"""scale_train.py — honest 50M/100M scale-up: ChaoticMixer vs Transformer.

Usage:  python scale_train.py [50|100] [chaos|tf] [steps]

Trains ONE model on corpus5m (5.2M tokens), evaluates PPL + order-3 beta-prior gate.
Both models use build_matched_pair() so param counts match within +-1% — fair fight.

Protocol (fixed for both):
  W=256, BATCH=64, LR=5e-4, WARMUP=1000, cosine, STEPS=8000, fp16 (AMP)
  + order-3 beta-prior gate (k=1.0) as in exp09/memory_selector
"""
import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parametric_models import build_matched_pair, count_params

VOCAB = 512
W = 256
ORDER = 3
BATCH = 64
LR = 5e-4
WARMUP = 1000
N_EVAL = 5000
EVAL_BATCH = 256


def load_corpus(name):
    p = os.path.join(HERE, f"corpus5m_{name}.txt")
    with open(p, encoding="utf-8", errors="ignore") as f:
        return f.read()


def make_bpe(text, vocab=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


def build_order3(train_ids):
    ctx_counts = defaultdict(dict)
    for i in range(ORDER, len(train_ids)):
        ctx = tuple(train_ids[i - ORDER + 1:i])
        w = train_ids[i]
        d_ = ctx_counts[ctx]
        d_[w] = d_.get(w, 0) + 1
    return ctx_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("budget", type=int, choices=[50, 100])
    ap.add_argument("kind", choices=["chaos", "tf"])
    ap.add_argument("steps", nargs="?", type=int, default=8000)
    args = ap.parse_args()

    print(f"Loading corpus5m...", flush=True)
    train_text = load_corpus("train")
    test_text = load_corpus("test")
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    train_ids = tok.encode(train_text).ids
    test_ids = tok.encode(test_text).ids
    print(f"  train_tokens={len(train_ids):,} test_tokens={len(test_ids):,}", flush=True)

    order3 = build_order3(train_ids)

    cm, tm, cl, tl = build_matched_pair(args.budget, V=V, W=W)
    model = cm if args.kind == "chaos" else tm
    n_params = count_params(model)
    print(f"[{args.budget}M/{args.kind}] params={n_params:,} layers={cl if args.kind=='chaos' else tl}", flush=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        p = (step - WARMUP) / max(1, args.steps - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    n = len(train_ids) - W - 1
    t0 = time.time()
    print(f"training {args.steps} steps...", flush=True)
    for s in range(args.steps):
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long, device=DEVICE)
        Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long, device=DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            loss = lossf(model(X), Y)
        scaler.scale(loss).backward()
        if s % 1000 == 0:
            g = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
            print(f"  [{s:,}] loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e} |g|={g:.2f} ({time.time()-t0:.0f}s)", flush=True)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

    print(f"trained ({time.time()-t0:.0f}s)", flush=True)
    # ---- eval ----
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
                X[k] = test_ids[i:i + W]
                y_te[s0 + k] = test_ids[i + W]
            out = model(torch.tensor(X, dtype=torch.long, device=DEVICE)).cpu().numpy()
            logits_te[s0:e] = out
    lpmix = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()
    mixer_only = float(np.exp(np.mean([-lpmix[k, y_te[k]] for k in range(N_EVAL)])))
    print(f"  {args.kind} alone PPL: {mixer_only:.3f}", flush=True)

    # order-3 beta-prior gate
    def eval_gated(kb):
        nll = np.zeros(N_EVAL)
        for k in range(N_EVAL):
            i = te_starts[k]
            pos = i + W
            ctx = tuple(test_ids[pos - ORDER + 1:pos])
            e = order3.get(ctx)
            pm = lpmix[k, y_te[k]]
            if e:
                tot = sum(e.values())
                c = e.get(int(y_te[k]), 0)
                if c > 0:
                    beta = tot / (tot + kb)
                    nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + np.log(c / tot))
                    continue
            nll[k] = -pm
        return float(np.exp(np.mean(nll)))

    gated = {kb: eval_gated(kb) for kb in [0.5, 1.0, 2.0]}
    print(f"  +order-3 beta-prior gate: {gated}", flush=True)

    out = {
        "budget_m": args.budget, "kind": args.kind, "params": n_params,
        "layers": cl if args.kind == "chaos" else tl, "steps": args.steps,
        "mixer_only": mixer_only, "gated": gated,
        "train_tokens": len(train_ids), "test_tokens": len(test_ids),
    }
    fname = f"scale_{args.budget}M_{args.kind}.json"
    json.dump(out, open(os.path.join(HERE, fname), "w"), indent=2)
    print(f"saved {fname}", flush=True)


if __name__ == "__main__":
    main()
