"""boost_morin2.py — oracle ceiling + match-length β (T3-lite) + Neural Trust (T3).

Questions:
  1. What is the ORACLE ceiling of prior boosting on this protocol?
  2. Does β that scales with match-LENGTH (not tot) beat fixed β=0.3?
  3. Does a learned β (logistic regression on features) beat fixed β?

Oracle variants (upper bounds, not deployable):
  O1: prior gives p=1 to the TRUE token when it matches (absolute ceiling)
  O2: prior = train-frequency of target (info-theoretic ceiling)
  O3: oracle chooses the BEST β per sample (ceiling of any β-scaling)

Deployable variants:
  T0: fixed β=0.3 + backoff (baseline morin)
  T1: adaptive β=tot/(tot+kb) (exp09 style)
  T3a: β from match length: β = L/(L+kb) — trust longer matches
  T3b: β from peakiness: β = p_max^γ — trust peaked priors
  T3c: Neural Trust — logistic regression on [match_len, log(tot), n_cand, p_max]
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
    return tok, tok.encode(tr).ids[:limit], tok.encode(te).ids


def build_prior(train_ids, max_order=8):
    tables = {L: defaultdict(Counter) for L in range(1, max_order + 1)}
    for i in range(1, len(train_ids)):
        tok = train_ids[i]
        for L in range(1, max_order + 1):
            if i - L < 0:
                break
            tables[L][tuple(train_ids[i - L:i])][tok] += 1
    return tables


def build_smoothed(train_ids, V, order=3):
    tables = {L: defaultdict(Counter) for L in range(1, order + 1)}
    for i in range(1, len(train_ids)):
        tok = train_ids[i]
        for L in range(1, order + 1):
            if i - L < 0:
                break
            tables[L][tuple(train_ids[i - L:i])][tok] += 1
    return tables


def smoothed_p(tables, ctx, target, V, lam):
    p = 0.0
    for L in range(1, len(tables) + 1):
        if len(ctx) < L:
            continue
        tab = tables[L].get(tuple(ctx[-L:]))
        if tab:
            st = sum(tab.values())
            p += lam[L - 1] * (tab.get(target, 0) + 1) / (st + V)
        else:
            p += lam[L - 1] * (1 / V)
    return p


def query_backoff(ctx, tables, max_order):
    best, best_tot = None, 0
    for L in range(min(max_order, len(ctx)), 0, -1):
        tab = tables[L].get(tuple(ctx[-L:]))
        if tab:
            tot = sum(tab.values())
            if tot > best_tot:
                best, best_tot = tab, tot
            break
    return best, best_tot


def prior_info(best, tot):
    """Peak probability + number of candidates for the matched prior."""
    if best is None:
        return 0.0, 0
    top = best.most_common(1)[0][1] / tot
    return top, len(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=500_000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--max-order", type=int, default=8)
    args = ap.parse_args()

    print("Loading...", flush=True)
    tok, train_ids, test_ids = get_ids(args.subsample)
    V = tok.get_vocab_size()
    priors = build_prior(train_ids, args.max_order)
    sm = build_smoothed(train_ids, V, order=3)
    lam = (0.2, 0.3, 0.5)
    print(f"V={V} train={len(train_ids):,} built", flush=True)

    rng = np.random.default_rng(42)
    n = len(test_ids) - args.max_order - 1
    idx = rng.choice(n, size=min(args.n_test, n), replace=False)

    rows = []  # per-sample record for analysis
    for i in idx:
        ctx = tuple(test_ids[max(0, i - args.max_order):i])
        target = test_ids[i]
        best, tot = query_backoff(ctx, priors, args.max_order)
        p_model = smoothed_p(sm, ctx, target, V, lam)
        # match length of the winning backoff
        mLen = 0
        if best is not None:
            for L in range(min(args.max_order, len(ctx)), 0, -1):
                if priors[L].get(tuple(ctx[-L:])):
                    mLen = L
                    break
        p_top, n_cand = prior_info(best, tot)
        c = best.get(target, 0) if best else 0
        p_prior = c / tot if tot > 0 else 0.0
        rows.append({
            "target": target, "ctx_len": len(ctx), "match_len": mLen,
            "log_tot": math.log(tot + 1), "p_top": p_top, "n_cand": n_cand,
            "p_model": p_model, "p_prior": p_prior, "prior_hit": tot > 0,
            "true_p_prior": p_prior,
        })

    # ---- metrics helper ----
    def nll_of(beta_fn):
        nll = 0.0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                nll += -np.log(pm)
                continue
            b = beta_fn(r)
            pp = r["p_prior"]
            mix = (1 - b) * pm + b * pp
            nll += -np.log(max(mix, 1e-300))
        return float(np.exp(nll / len(rows)))

    def acc_of(beta_fn):
        acc = 0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                pred = target_from_model(r)
            else:
                b = beta_fn(r)
                # argmax of mixture needs full prior; approximate with p_prior>0 test
                # use full prior distribution
                pred = argmax_mix(r, b)
            if pred == r["target"]:
                acc += 1
        return acc / len(rows)

    def target_from_model(r):
        # argmax of smoothed model = highest-order observed continuation
        return r["target"]  # placeholder (not used for PPL comparisons)

    def argmax_mix(r, b):
        ctx = None  # not available here; approximate acc via prior-correctness
        # compute from full prior if stored; here fallback: p_prior argmax
        return r["target"]  # placeholder — acc computed separately below

    # PPL metrics (honest, full)
    def ppl_fixed(beta):
        nll = 0.0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                nll += -np.log(pm); continue
            pp = r["p_prior"]
            nll += -np.log(max((1 - beta) * pm + beta * pp, 1e-300))
        return float(np.exp(nll / len(rows)))

    def ppl_adaptive(kb):
        nll = 0.0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                nll += -np.log(pm); continue
            tot = math.exp(r["log_tot"]) - 1
            b = tot / (tot + kb)
            pp = r["p_prior"]
            nll += -np.log(max((1 - b) * pm + b * pp, 1e-300))
        return float(np.exp(nll / len(rows)))

    def ppl_lenbeta(kb):
        nll = 0.0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                nll += -np.log(pm); continue
            b = r["match_len"] / (r["match_len"] + kb)
            pp = r["p_prior"]
            nll += -np.log(max((1 - b) * pm + b * pp, 1e-300))
        return float(np.exp(nll / len(rows)))

    def ppl_peak(gamma):
        nll = 0.0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                nll += -np.log(pm); continue
            b = min(0.9, r["p_top"] ** gamma)
            pp = r["p_prior"]
            nll += -np.log(max((1 - b) * pm + b * pp, 1e-300))
        return float(np.exp(nll / len(rows)))

    def ppl_oracle():
        """Oracle: best β per sample (upper bound of ANY β-scaling)."""
        nll = 0.0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                nll += -np.log(pm); continue
            pp = r["p_prior"]
            b = 1.0 if pp >= pm else 0.0
            nll += -np.log(max((1 - b) * pm + b * pp, 1e-300))
        return float(np.exp(nll / len(rows)))

    def ppl_oracle_peek():
        """Oracle: if true token has p>0 in prior, give it full prior mass."""
        nll = 0.0
        for r in rows:
            pm = r["p_model"]
            if not r["prior_hit"]:
                nll += -np.log(pm); continue
            if r["p_prior"] > 0:
                nll += -np.log(r["p_prior"])  # full prior confidence
            else:
                nll += -np.log(pm)
        return float(np.exp(nll / len(rows)))

    res = {
        "model_ppl": float(np.exp(np.mean([-np.log(r["p_model"]) for r in rows]))),
        "T0_fixed03": ppl_fixed(0.3),
        "T1_adaptive_kb1": ppl_adaptive(1.0),
        "T3a_lenbeta_kb2": ppl_lenbeta(2.0),
        "T3b_peak_gamma1": ppl_peak(1.0),
        "T3b_peak_gamma2": ppl_peak(2.0),
        "ORACLE_best_beta": ppl_oracle(),
        "ORACLE_prior_peek": ppl_oracle_peek(),
        "n_test": len(rows),
        "prior_coverage": np.mean([r["prior_hit"] for r in rows]),
        "mean_match_len": np.mean([r["match_len"] for r in rows]),
        "mean_prior_peak": np.mean([r["p_top"] for r in rows if r["prior_hit"]]),
        "prior_hit_true_rate": np.mean([r["true_p_prior"] > 0 for r in rows if r["prior_hit"]]),
    }
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v, 4)
    json.dump(res, open(os.path.join(HERE, "boost2_results.json"), "w"), indent=2)
    for k, v in res.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
