"""integrate.py — combine BOTH mechanisms: morin + NoteBook (static memory)
+ TrustScorer (learned head).

Architecture:
    morin prior (n-gram table)
        |
        +-- NoteBook (static memory: per-context "prior was right" verdicts)
        |        trust = smoothed right/(right+wrong)   -> feature
        |
        +-- TrustScorer (logistic head on static features)
                features = [match_len, log1p(tot), p_top, n_cand, ctx_len,
                            entropy, NOTEBOOK_TRUST]
                beta = sigmoid(w·features)   -> learned on val

Both signals feed ONE beta; the head learns how to weight them.

Variants evaluated on test:
  nb0  = morin fixed β=0.3            (baseline)
  nb1  = notebook trust only
  T3c  = head only (no notebook)
  INT  = head + notebook feature      (the integration)
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
sys.path.insert(0, r"C:\Users\Geroin\morin-filter")

from notebook_prior import (NotebookPrior, build_prior, build_smoothed,
                            backoff, model_logp_all)


def context_features(ctx, priors, max_order, nb, nb_weight=1.0):
    """Static features + notebook trust. Returns (feats, tab, tot) or (None,...)."""
    tab, tot = backoff(ctx, priors, max_order)
    if tab is None:
        return None, None, 0
    mLen = 0
    for L in range(min(max_order, len(ctx)), 0, -1):
        if priors[L].get(tuple(ctx[-L:])):
            mLen = L
            break
    p_top = max(tab.values()) / tot
    n_cand = len(tab)
    ps = np.array(list(tab.values()), dtype=np.float64) / tot
    entropy = float(-(ps * np.log(ps)).sum())
    nb_t = nb.trust(ctx)
    feats = np.array([mLen, math.log1p(tot), p_top, n_cand, len(ctx),
                      entropy, nb_t], dtype=np.float64)
    return feats, tab, tot


class IntegratedHead:
    """Logistic beta head with static features + notebook trust."""
    N_FEAT = 7

    def __init__(self):
        self.w = np.zeros(self.N_FEAT + 1)
        self._trained = False
        self._mean = None
        self._std = None

    def fit(self, X, pm, pp, lr=0.03, steps=3000):
        X = np.array(X, dtype=np.float64)
        pm = np.array(pm)
        pp = np.array(pp)
        self._mean = X.mean(0)
        self._std = X.std(0)
        self._std = np.where(self._std < 1e-6, 1.0, self._std)
        Xs = (X - self._mean) / self._std
        w = np.zeros(Xs.shape[1] + 1)
        best = None
        for _ in range(steps):
            z = Xs @ w[:-1] + w[-1]
            b = 1.0 / (1.0 + np.exp(-z))
            b = np.clip(b, 0.05, 0.95)
            mix = (1 - b) * pm + b * pp
            d = -(pp - pm) / np.clip(mix, 1e-300, None)
            g_beta = d * b * (1 - b)
            grad = np.concatenate([[g_beta.sum()], Xs.T @ g_beta])
            w -= lr * grad / len(pm)
            nll = np.mean(-np.log(np.clip(mix, 1e-300, 1)))
            if best is None or nll < best[0]:
                best = (nll, w.copy())
        self.w = best[1]
        self._trained = True
        return best[0]

    def beta(self, feats):
        if not self._trained:
            return 0.3
        x = (np.array(feats) - self._mean) / self._std
        z = x @ self.w[:-1] + self.w[-1]
        b = 1.0 / (1.0 + math.exp(-z))
        return max(0.05, min(b, 0.95))


def collect(ids, idx, priors, max_order, sm, V, lam, nb):
    rows = []
    for i in idx:
        ctx = tuple(ids[max(0, i - max_order):i])
        target = ids[i]
        feats, tab, tot = context_features(ctx, priors, max_order, nb)
        lp = model_logp_all(sm, ctx, V, lam)
        prior_right = bool(tab and tab.get(target, 0) > 0)
        rows.append((ctx, target, lp, tab, tot, feats, prior_right))
    return rows


def eval_variants(rows, priors, max_order, head, nb):
    nll = {"nb0": 0.0, "nb1": 0.0, "T3c": 0.0, "INT": 0.0}
    acc = {"nb0": 0, "nb1": 0, "T3c": 0, "INT": 0}
    V = None
    for ctx, target, lp, tab, tot, feats, prior_right in rows:
        V = len(lp)
        b0 = 0.3
        b1 = min(0.95, max(0.05, nb.trust(ctx)))
        # T3c: head WITHOUT notebook feature (zero it out)
        if feats is not None:
            f_no_nb = feats.copy()
            f_no_nb[-1] = 0.5   # neutral notebook trust
            b3 = head.beta(f_no_nb)
            b4 = head.beta(feats)
        else:
            b3 = b4 = b0
        for key, b in (("nb0", b0), ("nb1", b1), ("T3c", b3), ("INT", b4)):
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
    n = len(rows)
    return {k: (float(np.exp(v / n)), acc[k] / n) for k, v in nll.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["5m", "stack"], default="5m")
    ap.add_argument("--stack-mb", type=int, default=20)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

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

    print("Building prior + smoothed model...", flush=True)
    t0 = time.time()
    priors = build_prior(prior_ids, args.max_order)
    sm = build_smoothed(model_ids, V, order=3)
    lam = (0.2, 0.3, 0.5)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    print("Building STATIC notebook from train...", flush=True)
    nb = NotebookPrior(max_entries=200_000)
    step = max(1, len(prior_ids) // 200_000)
    for i in range(args.max_order, len(prior_ids), step):
        ctx = tuple(prior_ids[max(0, i - args.max_order):i])
        target = prior_ids[i]
        tab, tot = backoff(ctx, priors, args.max_order)
        prior_right = bool(tab and tab.get(target, 0) > 0)
        nb.update(ctx, prior_right)
    print(f"  entries={nb.size():,}", flush=True)

    rng = np.random.default_rng(42)
    n = len(test_ids) - args.max_order - 1
    all_idx = rng.choice(n, size=min(args.n_test * 2, n), replace=False)
    half = len(all_idx) // 2
    val_rows = collect(test_ids, all_idx[:half], priors, args.max_order, sm, V, lam, nb)
    te_rows = collect(test_ids, all_idx[half:], priors, args.max_order, sm, V, lam, nb)
    print(f"val={len(val_rows)} test={len(te_rows)}", flush=True)

    # fit integrated head on val
    X, pm, pp = [], [], []
    for ctx, target, lp, tab, tot, feats, prior_right in val_rows:
        if feats is None:
            continue
        c = tab.get(target, 0)
        X.append(feats)
        pm.append(float(np.exp(float(lp[target]))))
        pp.append(c / tot)
    head = IntegratedHead()
    nll = head.fit(X, pm, pp, lr=0.03, steps=3000)
    print(f"  head fit NLL={nll:.4f}  w={np.round(head.w, 3)}", flush=True)

    res = eval_variants(te_rows, priors, args.max_order, head, nb)
    print(f"\n=== INTEGRATION [{tag}] ===")
    for key, (ppl, a) in res.items():
        print(f"  {key}: PPL={ppl:.3f}  acc={a:.3f}")
    out = {"tag": tag, "results": {k: {"ppl": p, "acc": a} for k, (p, a) in res.items()},
           "head_w": head.w.tolist(), "notebook_entries": nb.size()}
    json.dump(out, open(os.path.join(HERE, f"integrate_{tag}.json"), "w"), indent=2)
    print("saved integrate json")


if __name__ == "__main__":
    main()
