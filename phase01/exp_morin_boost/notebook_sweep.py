"""notebook_sweep.py — Step 2: memory vs quality curve for NoteBook Prior.

One prior build, then sweep max_entries: [0 (no notebook), 100, 1K, 10K, 50K,
200K]. Shows how much memory the notebook needs for its gain.

Usage: python notebook_sweep.py [--corpus 5m] [--n-test 10000]
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict, Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, r"C:\Users\Geroin\morin-filter")

from notebook_prior import (NotebookPrior, build_prior, build_smoothed,
                            backoff, model_logp_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["5m", "stack"], default="5m")
    ap.add_argument("--stack-mb", type=int, default=20)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--entries", type=str, default="0,100,1000,10000,50000,200000")
    args = ap.parse_args()
    entries_list = [int(x) for x in args.entries.split(",")]

    from exp_memory_selector.experiment import build_tokenizer
    tok = build_tokenizer()
    V = tok.get_vocab_size()

    if args.corpus == "5m":
        with open(os.path.join(PHASE, "corpus5m_train.txt"), encoding="utf-8", errors="ignore") as f:
            tr = f.read()
        with open(os.path.join(PHASE, "corpus5m_test.txt"), encoding="utf-8", errors="ignore") as f:
            te = f.read()
        model_ids = tok.encode(tr).ids
        prior_ids = model_ids
        test_ids = tok.encode(te).ids
        tag = "corpus5m"
    else:
        with open(os.path.join(PHASE, "corpus_stack_train.txt"), encoding="utf-8", errors="ignore") as f:
            st = f.read(args.stack_mb * 1_000_000)
        stack_ids = tok.encode(st).ids
        n_all = len(stack_ids)
        model_ids = stack_ids[:int(n_all * 0.3)]
        prior_ids = stack_ids[int(n_all * 0.3):int(n_all * 0.8)]
        test_ids = stack_ids[int(n_all * 0.8):]
        tag = f"stack{args.stack_mb}mb"
    print(f"[{tag}] V={V} test={len(test_ids):,}", flush=True)

    print("Building prior + smoothed model (once)...", flush=True)
    t0 = time.time()
    priors = build_prior(prior_ids, args.max_order)
    sm = build_smoothed(model_ids, V, order=3)
    lam = (0.2, 0.3, 0.5)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(42)
    n = len(test_ids) - args.max_order - 1
    idx = rng.choice(n, size=min(args.n_test, n), replace=False)

    # precompute per-position data once
    print("Precomputing positions...", flush=True)
    samples = []
    for i in idx:
        ctx = tuple(test_ids[max(0, i - args.max_order):i])
        target = test_ids[i]
        tab, tot = backoff(ctx, priors, args.max_order)
        lp = model_logp_all(sm, ctx, V, lam)
        prior_right = bool(tab and tab.get(target, 0) > 0)
        mLen = 0
        if tab:
            for L in range(min(args.max_order, len(ctx)), 0, -1):
                if priors[L].get(tuple(ctx[-L:])):
                    mLen = L
                    break
        samples.append((ctx, target, lp, tab, tot, prior_right, mLen))

    results = {}
    for me in entries_list:
        nb = NotebookPrior(max_entries=me)
        nll = {"nb0": 0.0, "nb1": 0.0, "nb2": 0.0}
        acc = {"nb0": 0, "nb1": 0, "nb2": 0}
        for pos, (ctx, target, lp, tab, tot, prior_right, mLen) in enumerate(samples):
            b0 = 0.3
            b1 = min(0.95, max(0.05, nb.trust(ctx)))
            if tab:
                b2 = min(0.95, max(0.05, 0.5 * nb.trust(ctx) + 0.5 * (mLen / args.max_order)))
            else:
                b2 = b0
            for key, b in (("nb0", b0), ("nb1", b1), ("nb2", b2)):
                if tab and tot > 0:
                    prior_logp = np.full(V, -np.inf, dtype=np.float64)
                    for tok, c in tab.items():
                        prior_logp[tok] = np.log(c / tot)
                    with np.errstate(divide="ignore"):
                        mix = np.logaddexp(np.log1p(-b) + lp, np.log(b) + prior_logp)
                else:
                    mix = lp
                nll[key] += -mix[target]
                if int(np.argmax(mix)) == target:
                    acc[key] += 1
            if pos >= args.warmup:
                nb.update(ctx, prior_right)
        n_s = len(samples)
        results[me] = {
            "ppl": {k: float(np.exp(v / n_s)) for k, v in nll.items()},
            "acc": {k: acc[k] / n_s for k in acc},
            "entries": nb.size(),
        }
        print(f"  max_entries={me}: entries={nb.size():,} "
              f"nb1={results[me]['ppl']['nb1']:.3f} "
              f"nb2={results[me]['ppl']['nb2']:.3f} "
              f"nb0={results[me]['ppl']['nb0']:.3f}", flush=True)

    json.dump({"tag": tag, "results": results},
              open(os.path.join(HERE, f"notebook_sweep_{tag}.json"), "w"), indent=2)
    print("saved notebook_sweep json")


if __name__ == "__main__":
    main()
