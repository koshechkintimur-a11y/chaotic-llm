"""stack_trust.py — scale test: does TrustScorer win GROW with corpus size?

Builds the n-gram prior on a subset of the 503M-token Stack code corpus, fits
TrustScorer, and measures PPL gain over fixed-β morin on the SAME 5M-token
model. Shows whether the trainable prior scales with data.

Usage: python stack_trust.py [--stack-mb 80] [--max-order 8] [--n-test 5000]
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

from morin_filter import build_ngram_prior, CorpusPrior, TrustScorer


def load_tokens():
    """Return (tok, train5_ids, stack_ids, test_ids)."""
    from exp_memory_selector.experiment import build_tokenizer
    tok = build_tokenizer()
    with open(os.path.join(PHASE, "corpus5m_train.txt"), encoding="utf-8", errors="ignore") as f:
        tr5m = f.read()
    with open(os.path.join(PHASE, "corpus5m_test.txt"), encoding="utf-8", errors="ignore") as f:
        te = f.read()
    train5 = tok.encode(tr5m).ids
    test_ids = tok.encode(te).ids
    return tok, train5, test_ids


def build_prior(ids, max_order):
    tables = {L: defaultdict(Counter) for L in range(1, max_order + 1)}
    for i in range(1, len(ids)):
        tok = ids[i]
        for L in range(1, max_order + 1):
            if i - L < 0:
                break
            tables[L][tuple(ids[i - L:i])][tok] += 1
    return tables


def build_smoothed(ids, V, order=3):
    tables = {L: defaultdict(Counter) for L in range(1, order + 1)}
    for i in range(1, len(ids)):
        tok = ids[i]
        for L in range(1, order + 1):
            if i - L < 0:
                break
            tables[L][tuple(ids[i - L:i])][tok] += 1
    return tables


def model_logp_all(tables, ctx, V, lam):
    lp = np.full(V, np.log(1 / V), dtype=np.float64)
    for L in range(1, len(tables) + 1):
        if len(ctx) < L:
            continue
        tab = tables[L].get(tuple(ctx[-L:]))
        if tab:
            st = sum(tab.values())
            for t, c in tab.items():
                lp[t] = np.logaddexp(lp[t], np.log(lam[L - 1] * (c + 1) / (st + V)))
    lp -= np.logaddexp.reduce(lp)
    return lp


def collect(ids, idx, sm, V, lam, max_order):
    ctxs, tgts, lps = [], [], []
    for i in idx:
        ctx = tuple(ids[max(0, i - max_order):i])
        ctxs.append(ctx)
        tgts.append(ids[i])
        lps.append(model_logp_all(sm, ctx, V, lam))
    return ctxs, tgts, lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-mb", type=int, default=40)   # MB of stack text to read
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=5000)
    args = ap.parse_args()

    tok, train5, test_ids = load_tokens()
    V = tok.get_vocab_size()
    print(f"V={V} train5={len(train5):,} test={len(test_ids):,}", flush=True)

    # read stack text (bounded) — ALL three splits from Stack (honest scale test)
    stack_path = os.path.join(PHASE, "corpus_stack_train.txt")
    n_bytes = args.stack_mb * 1_000_000
    with open(stack_path, encoding="utf-8", errors="ignore") as f:
        stack_text = f.read(n_bytes)
    stack_ids = tok.encode(stack_text).ids
    # 3-way split by POSITION: 30% model, 50% prior, 20% test (all Stack)
    n_all = len(stack_ids)
    n_model = int(n_all * 0.3)
    n_prior = int(n_all * 0.5)
    model_ids = stack_ids[:n_model]
    prior_ids = stack_ids[n_model:n_model + n_prior]
    test_ids = stack_ids[n_model + n_prior:]
    print(f"stack read: {args.stack_mb}MB -> {len(stack_ids):,} tokens")
    print(f"  model: {len(model_ids):,} | prior: {len(prior_ids):,} | "
          f"test: {len(test_ids):,}", flush=True)

    print("Building prior from Stack...", flush=True)
    t0 = time.time()
    tables = build_prior(prior_ids, args.max_order)
    flat = {}
    for L in range(1, args.max_order + 1):
        for k, v in tables[L].items():
            flat[k] = dict(v)
    prior = CorpusPrior(flat)
    print(f"  {time.time()-t0:.0f}s, contexts={len(flat):,}", flush=True)

    print("Building smoothed model from Stack (30% split)...", flush=True)
    sm = build_smoothed(model_ids, V, order=3)
    lam = (0.2, 0.3, 0.5)

    rng = np.random.default_rng(42)
    n = len(test_ids) - args.max_order - 1
    all_idx = rng.choice(n, size=min(args.n_test * 2, n), replace=False)
    half = len(all_idx) // 2
    v_ctx, v_tgt, v_lp = collect(test_ids, all_idx[:half], sm, V, lam, args.max_order)
    t_ctx, t_tgt, t_lp = collect(test_ids, all_idx[half:], sm, V, lam, args.max_order)
    print(f"val={len(v_ctx)} test={len(t_ctx)}", flush=True)

    ts = TrustScorer(prior, beta_fallback=0.3, max_back=args.max_order)
    nll = ts.fit_batch(v_ctx, v_tgt, v_lp, lr=0.03, steps=2000)
    print(f"  val NLL={nll:.4f}  w={np.round(ts.w,3)}", flush=True)

    morin0 = TrustScorer(prior, beta_fallback=0.3, max_back=args.max_order)

    def eval_scorer(sc):
        nll = 0.0
        acc = 0
        for ctx, tgt, lp in zip(t_ctx, t_tgt, t_lp):
            mix = sc.mixture_logp(lp, ctx)
            nll += -mix[tgt]
            if int(np.argmax(mix)) == tgt:
                acc += 1
        return float(np.exp(nll / len(t_ctx))), acc / len(t_ctx)

    base = float(np.exp(np.mean([-lp[t] for lp, t in zip(t_lp, t_tgt)])))
    morin_ppl, morin_acc = eval_scorer(morin0)
    trust_ppl, trust_acc = eval_scorer(ts)
    print(f"\n  model:  {base:.3f}")
    print(f"  morin:  {morin_ppl:.3f} acc={morin_acc:.3f}  ({base/morin_ppl:.2f}x)")
    print(f"  trust:  {trust_ppl:.3f} acc={trust_acc:.3f}  ({base/trust_ppl:.2f}x, "
          f"{morin_ppl/trust_ppl:.2f}x vs morin)")

    res = {"stack_mb": args.stack_mb, "model": base, "morin": morin_ppl,
           "trust": trust_ppl, "morin_acc": morin_acc, "trust_acc": trust_acc}
    json.dump(res, open(os.path.join(HERE, "stack_trust_results.json"), "w"), indent=2)
    print("saved stack_trust_results.json")


if __name__ == "__main__":
    main()
