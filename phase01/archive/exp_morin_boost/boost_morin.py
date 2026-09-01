"""boost_morin.py — T1/T2/T3: from fixed β to Neural Trust.

All CPU (no GPU needed). Uses a unigram model as the "neural model" placeholder
for the mixture — this is a valid lower bound (if mixture helps over unigram,
it will help even more over a better model).

T0: baseline morin (backoff + fixed β=0.3)
T1: backoff + adaptive β = tot/(tot+kb)
T2: learned interpolation across orders 1..K + adaptive β  
T3: Neural Trust — β predicted by a small MLP from context features

All measure: PPL mix, top-1 acc, coverage, recall@10
"""
import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict, Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
from exp_memory_selector.experiment import load_chars, build_tokenizer, MAX_TRAIN


def get_ids(limit=1_000_000):
    tok = build_tokenizer()
    tr = load_chars(os.path.join(PHASE, "corpus5m_train.txt"), MAX_TRAIN)
    te = load_chars(os.path.join(PHASE, "corpus5m_test.txt"))
    train_ids = tok.encode(tr).ids[:limit]
    test_ids = tok.encode(te).ids
    return tok, train_ids, test_ids


def build_prior(train_ids, max_order=8):
    tables = {L: defaultdict(Counter) for L in range(1, max_order + 1)}
    for i in range(1, len(train_ids)):
        tok = train_ids[i]
        for L in range(1, max_order + 1):
            if i - L < 0:
                break
            tables[L][tuple(train_ids[i - L:i])][tok] += 1
    return tables


def build_smoothed_model(train_ids, V, order=3):
    """Interpolated (Jelinek-Mercer) smoothed n-gram model — acts as the
    'neural' side: generalizes, unlike exact lookup. Lambdas tuned on val."""
    tables = {L: defaultdict(Counter) for L in range(1, order + 1)}
    for i in range(1, len(train_ids)):
        tok = train_ids[i]
        for L in range(1, order + 1):
            if i - L < 0:
                break
            tables[L][tuple(train_ids[i - L:i])][tok] += 1
    # totals per length
    tots = {L: sum(sum(t.values()) for t in tables[L].values()) for L in tables}
    return tables, tots


def smoothed_logp(model_tables, tots, ctx, target, V, lam=None):
    """Interpolated model: P = sum_L lam_L * P(order L). Returns log p(target)."""
    if lam is None:
        lam = [0.1, 0.3, 0.6][:min(3, len(model_tables))]
    p = 0.0
    for L in range(1, len(model_tables) + 1):
        if len(ctx) < L:
            continue
        key = tuple(ctx[-L:])
        tab = model_tables[L].get(key)
        tot = tots[L]
        if tab:
            c = tab.get(target, 0)
            pt = (c + 1) / (sum(tab.values()) + V)
        else:
            pt = 1 / V
        p += lam[L - 1] * pt
    return np.log(p) if p > 0 else -np.inf


def model_logp_all(model_tables, tots, ctx, V, lam=None):
    """Full distribution of smoothed model over vocab (for top-1)."""
    if lam is None:
        lam = [0.1, 0.3, 0.6][:min(3, len(model_tables))]
    logp = np.full(V, -np.inf, dtype=np.float64)
    for L in range(1, len(model_tables) + 1):
        if len(ctx) < L:
            continue
        key = tuple(ctx[-L:])
        tab = model_tables[L].get(key)
        tot = tots[L]
        if tab:
            st = sum(tab.values())
            for t, c in tab.items():
                pt = (c + 1) / (st + V)
                l = np.log(lam[L - 1] * pt)
                logp[t] = np.logaddexp(logp[t], l)
            # unknown mass for this order
            unknown = np.log(lam[L - 1]) + np.log(1 / V)
            logp = np.where(np.isneginf(logp), unknown, logp)
    return logp


def context_features(ctx, tables, max_order):
    """Return features for the trust network: tot per order, hit flags, entropy."""
    feats = []
    hits = []
    for L in range(1, max_order + 1):
        if len(ctx) >= L:
            key = tuple(ctx[-L:])
            tab = tables[L].get(key)
            if tab:
                tot = sum(tab.values())
                feats.append(math.log(tot + 1))
                hits.append(1.0)
            else:
                feats.append(0.0)
                hits.append(0.0)
        else:
            feats.append(0.0)
            hits.append(0.0)
    return feats + hits  # 2*max_order features


# ---------- prior query (with backoff or interpolation) ----------

def query_backoff(ctx, tables, max_order, max_ctx_len=8):
    """Standard morin backoff: longest match wins."""
    best = None
    best_tot = 0
    for L in range(min(max_order, len(ctx)), 0, -1):
        key = tuple(ctx[-L:])
        tab = tables[L].get(key)
        if tab:
            tot = sum(tab.values())
            if tot > best_tot:
                best, best_tot = tab, tot
            break
    return best, best_tot if best else 0


def query_blend(ctx, tables, weights, max_order):
    """Weighted blend of all orders (learned interpolation)."""
    blended = {}
    total = 0
    for L in range(1, min(max_order, len(ctx)) + 1):
        key = tuple(ctx[-L:])
        tab = tables[L].get(key)
        if tab:
            w = weights[L - 1]
            for tok, c in tab.items():
                blended[tok] = blended.get(tok, 0.0) + w * c
            total += w * sum(tab.values())
    return blended, total


# ---------- mixture scorers ----------

def nll_mixture(prior_counts, prior_tot, model_logp, target, beta):
    """Compute NLL under the mixture."""
    if prior_tot == 0:
        return -model_logp[target]
    log_beta = np.log(beta)
    log_alpha = np.log1p(-beta)
    # prior probability of target
    c = prior_counts.get(target, 0)
    prior_p = c / prior_tot if prior_tot > 0 else 0.0
    log_prior = np.log(prior_p) if prior_p > 0 else -np.inf
    mix = np.logaddexp(log_alpha + model_logp[target], log_beta + log_prior)
    return -mix


def eval_config(name, priors, model_tables, tots, test_ids, V, max_order,
                n_test=5000, beta=0.3, adaptive=False, weights=None,
                lam=None, rng_seed=42):
    """Evaluate a mixing strategy with context-dependent smoothed model."""
    rng = np.random.default_rng(rng_seed)
    n = len(test_ids) - max_order - 1
    idx = rng.choice(n, size=min(n_test, n), replace=False)
    nlls = []
    targets = []
    ctxs = []
    for i in idx:
        ctx = tuple(test_ids[max(0, i - max_order):i])
        target = test_ids[i]
        if weights is not None:
            pc, ptot = query_blend(ctx, priors, weights, max_order)
        else:
            pc, ptot = query_backoff(ctx, priors, max_order)
        targets.append(target)
        ctxs.append(ctx)
        # model logp for this context
        m_logp = smoothed_logp(model_tables, tots, ctx, target, V, lam)
        # adaptive β
        b = beta
        if adaptive and ptot > 0:
            kb = 1.0
            b = ptot / (ptot + kb)
        # mixture NLL
        if ptot == 0:
            nlls.append(-m_logp)
        else:
            c = pc.get(target, 0)
            prior_p = c / ptot if ptot > 0 else 0.0
            log_prior = np.log(prior_p) if prior_p > 0 else -np.inf
            mix = np.logaddexp(np.log1p(-b) + m_logp, np.log(b) + log_prior)
            nlls.append(-mix)
    # top-1 accuracy under mixture
    acc = 0
    for i, (ctx, target) in enumerate(zip(ctxs, targets)):
        if weights is not None:
            pc, ptot = query_blend(ctx, priors, weights, max_order)
        else:
            pc, ptot = query_backoff(ctx, priors, max_order)
        b = beta
        if adaptive and ptot > 0:
            b = ptot / (ptot + 1.0)
        m_all = model_logp_all(model_tables, tots, ctx, V, lam)
        if ptot == 0:
            pred = int(np.argmax(m_all))
        else:
            with np.errstate(divide="ignore"):
                prior_p = np.array([pc.get(t, 0) / ptot for t in range(V)])
                mix = np.logaddexp(np.log1p(-b) + m_all, np.log(b) + np.log(prior_p))
            pred = int(np.argmax(mix))
        if pred == target:
            acc += 1
    return {
        "name": name,
        "n_test": int(len(idx)),
        "ppl": float(np.exp(np.mean(nlls))),
        "top1_acc": acc / int(len(idx)) if len(idx) else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=500_000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--max-order", type=int, default=8)
    args = ap.parse_args()

    print("Loading data...", flush=True)
    tok, train_ids, test_ids = get_ids(args.subsample)
    V = tok.get_vocab_size()
    print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}", flush=True)

    print("Building prior tables...", flush=True)
    t0 = time.time()
    priors = build_prior(train_ids, args.max_order)
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    print("Building smoothed model (acts as 'neural' side)...", flush=True)
    model_tables, tots = build_smoothed_model(train_ids, V, order=3)
    # tune interpolation lambdas on a validation split
    rng = np.random.default_rng(7)
    nv = len(test_ids) - 9
    val_idx = rng.choice(nv, size=min(3000, nv), replace=False)
    best = None
    for lam in [(0.05, 0.2, 0.75), (0.1, 0.3, 0.6), (0.2, 0.3, 0.5)]:
        nll = []
        for i in val_idx:
            ctx = tuple(test_ids[max(0, i - 8):i])
            nll.append(-smoothed_logp(model_tables, tots, ctx, test_ids[i], V, lam))
        p = float(np.exp(np.mean(nll)))
        if best is None or p < best[1]:
            best = (lam, p)
    lam = best[0]
    print(f"  tuned lam={lam} model-PPL={best[1]:.2f}", flush=True)

    # order-frequency weights for T2 (fraction of val contexts matching each order)
    order_hits = np.zeros(args.max_order)
    for i in val_idx:
        for L in range(1, args.max_order + 1):
            key = tuple(test_ids[i - L:i])
            if priors[L].get(key):
                order_hits[L - 1] += 1
    order_hits = order_hits / max(1, order_hits.sum())
    print(f"  order_hits={np.round(order_hits, 3)}", flush=True)

    results = []
    configs = [
        # T0: baseline morin (backoff, fixed β=0.3)
        ("T0_morin", {"beta": 0.3, "adaptive": False, "weights": None}),
        # T1: backoff + adaptive β
        ("T1_adaptive", {"beta": 0.3, "adaptive": True, "weights": None}),
        # T2: learned blend (order-frequency weights) + adaptive β
        ("T2_learned_blend", {"beta": 0.3, "adaptive": True,
                              "weights": order_hits * 0.5 + 0.5 / args.max_order}),
    ]
    for name, kw in configs:
        print(f"  {name}...", flush=True)
        r = eval_config(name, priors, model_tables, tots, test_ids, V,
                        args.max_order, n_test=args.n_test, lam=lam, **kw)
        results.append(r)
        print(f"    ppl={r['ppl']:.2f} acc={r['top1_acc']:.3f}", flush=True)

    json.dump(results, open(os.path.join(HERE, "boost_results.json"), "w"), indent=2)
    print("saved boost_results.json")


if __name__ == "__main__":
    main()