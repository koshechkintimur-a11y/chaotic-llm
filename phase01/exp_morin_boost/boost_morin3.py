"""boost_morin3.py — T3c: Neural Trust (learned β from features).

Honest protocol:
  - collect per-sample rows (features + outcome) on val split
  - train a small logistic head: β = sigmoid(w·f) minimizing NLL
  - apply to test split (never seen by the head)
  - compare: T0 (β=0.3), T3a (β=L/(L+kb)), T3c (learned), ORACLE

Features per sample: [match_len, log(1+tot), p_top, n_cand, ctx_len]
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


def get_ids(limit=5_000_000):
    tok = build_tokenizer()
    # bypass MAX_TRAIN: load full corpus
    train_path = os.path.join(PHASE, "corpus5m_train.txt")
    test_path = os.path.join(PHASE, "corpus5m_test.txt")
    with open(train_path, encoding="utf-8", errors="ignore") as f:
        tr = f.read()
    with open(test_path, encoding="utf-8", errors="ignore") as f:
        te = f.read()
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
    best, best_tot, best_L = None, 0, 0
    for L in range(min(max_order, len(ctx)), 0, -1):
        tab = tables[L].get(tuple(ctx[-L:]))
        if tab:
            tot = sum(tab.values())
            if tot > best_tot:
                best, best_tot, best_L = tab, tot, L
            break
    return best, best_tot, best_L


def collect_rows(tables, sm, ids, V, max_order, lam, idx):
    rows = []
    for i in idx:
        ctx = tuple(ids[max(0, i - max_order):i])
        target = ids[i]
        best, tot, mLen = query_backoff(ctx, tables, max_order)
        p_model = smoothed_p(sm, ctx, target, V, lam)
        p_top, n_cand = 0.0, 0
        if best is not None:
            p_top = best.most_common(1)[0][1] / tot
            n_cand = len(best)
        c = best.get(target, 0) if best else 0
        p_prior = c / tot if tot > 0 else 0.0
        feats = [mLen, math.log(tot + 1), p_top, n_cand, len(ctx)]
        rows.append({
            "feats": feats, "p_model": p_model, "p_prior": p_prior,
            "prior_hit": tot > 0, "target": target,
        })
    return rows


def nll_all(rows, beta_fn):
    nll = 0.0
    for r in rows:
        pm = r["p_model"]
        if not r["prior_hit"]:
            nll += -np.log(pm)
            continue
        b = beta_fn(r)
        pp = r["p_prior"]
        nll += -np.log(max((1 - b) * pm + b * pp, 1e-300))
    return float(np.exp(nll / len(rows)))


def train_trust(rows_train, lr=0.03, steps=3000):
    """Train logistic β head: b = sigmoid(w·f + b0), minimize NLL.
    Features standardized with robust scale (std floor)."""
    X = np.array([r["feats"] for r in rows_train], dtype=np.float64)
    pm = np.array([r["p_model"] for r in rows_train])
    pp = np.array([r["p_prior"] for r in rows_train])
    hit = np.array([r["prior_hit"] for r in rows_train])
    X = X[hit]
    pm = pm[hit]
    pp = pp[hit]
    mean = X.mean(0)
    std = X.std(0)
    std = np.where(std < 1e-6, 1.0, std)   # avoid div-by-zero on constant features
    Xs = (X - mean) / std
    # augment with p_top explicitly (feature idx 2) — already in X
    w = np.zeros(Xs.shape[1] + 1)
    best = None
    for step in range(steps):
        logit = Xs @ w[1:] + w[0]
        b = 1 / (1 + np.exp(-logit))
        b = np.clip(b, 0.05, 0.95)
        mix = (1 - b) * pm + b * pp
        # NLL gradient (already correct sign: db = dNLL/db)
        d_mix = pp - pm
        db = -d_mix / mix
        db = db * b * (1 - b)
        g = np.concatenate([[db.sum()], Xs.T @ db])
        w -= lr * g / len(pm)
        if step % 500 == 0 or step == steps - 1:
            nll = np.mean(-np.log(np.clip(mix, 1e-300, 1)))
            if best is None or nll < best[0]:
                best = (nll, w.copy())
    return best[1], mean, std


def beta_learned(feats, w, mean, std):
    x = (np.array(feats) - mean) / (std + 1e-9)
    logit = x @ w[1:] + w[0]
    return float(1 / (1 + np.exp(-logit)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=5_000_000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--max-order", type=int, default=8)
    args = ap.parse_args()

    print("Loading...", flush=True)
    tok, train_ids, test_ids = get_ids(args.subsample)
    V = tok.get_vocab_size()
    priors = build_prior(train_ids, args.max_order)
    sm = build_smoothed(train_ids, V, order=3)
    lam = (0.2, 0.3, 0.5)
    print(f"V={V} train={len(train_ids):,}", flush=True)

    rng = np.random.default_rng(42)
    n = len(test_ids) - args.max_order - 1
    all_idx = rng.choice(n, size=min(args.n_test * 2, n), replace=False)
    half = len(all_idx) // 2
    val_idx, te_idx = all_idx[:half], all_idx[half:]

    rows_val = collect_rows(priors, sm, test_ids, V, args.max_order, lam, val_idx)
    rows_te = collect_rows(priors, sm, test_ids, V, args.max_order, lam, te_idx)
    print(f"val={len(rows_val)} test={len(rows_te)}", flush=True)

    # train trust head on val
    w, mean, std = train_trust(rows_val)
    print(f"  w={np.round(w, 3)}", flush=True)

    def b_learned(r):
        return beta_learned(r["feats"], w, mean, std)

    def b_fixed(r):
        return 0.3

    def b_len(r):
        L = r["feats"][0]
        return L / (L + 2.0)

    def b_oracle(r):
        return 1.0 if r["p_prior"] >= r["p_model"] else 0.0

    results = {
        "n_test": len(rows_te),
        "model_ppl": float(np.exp(np.mean([-np.log(r["p_model"]) for r in rows_te]))),
        "T0_fixed03": nll_all(rows_te, b_fixed),
        "T3a_lenbeta": nll_all(rows_te, b_len),
        "T3c_learned": nll_all(rows_te, b_learned),
        "ORACLE": nll_all(rows_te, b_oracle),
    }
    for k, v in results.items():
        if isinstance(v, float):
            results[k] = round(v, 4)
    json.dump(results, open(os.path.join(HERE, "boost3_results.json"), "w"), indent=2)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
