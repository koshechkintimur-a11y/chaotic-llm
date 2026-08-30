"""notebook_prior.py — NoteBook Prior: regulator keeps a running reference book
of "was the prior right in this context" verdicts, and uses it to set β.

Architect's idea: instead of GENERALIZING when to trust the prior (static head,
fails on diverse Stack code), MEMORIZE concrete verdicts per context. At each
step we know the real next token (ground truth) — we update the notebook; for
a new context we look up similar verdicts (O(1) hash) and set β accordingly.

This is a recency-style signal (like Cache LM / induction heads): the notebook
accumulates DURING the pass and informs SUBSEQUENT predictions. No leakage:
each step uses only verdicts recorded BEFORE it.

Two notebook policies:
  NB-1: β from notebook only
  NB-2: β from notebook + static features (hybrid)
Compared against NB-0: morin fixed β=0.3.
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


class NotebookPrior:
    """Running reference book: context-key -> [right, wrong] verdicts about
    whether the corpus prior contained the true next token."""

    def __init__(self, max_entries: int = 500_000):
        self.notes = defaultdict(lambda: [0, 0])
        self.max_entries = max_entries

    def key(self, ctx: tuple) -> tuple:
        """Key = longest observed prior context for this window (like morin)."""
        # use last up to 8 tokens, truncated to what the prior knows
        return ctx[-8:]

    def lookup(self, ctx: tuple):
        """(right, wrong) counts for this context (or empty)."""
        return self.notes.get(self.key(ctx), (0, 0))

    def trust(self, ctx: tuple, alpha: float = 1.0) -> float:
        """Smoothed fraction of times the prior was right for this context."""
        r, w = self.lookup(ctx)
        if r + w == 0:
            return 0.5
        return (r + alpha) / (r + w + 2 * alpha)

    def update(self, ctx: tuple, target_in_prior: bool) -> None:
        """Record a verdict: did the prior contain the true next token?"""
        k = self.key(ctx)
        if len(self.notes) < self.max_entries:
            if target_in_prior:
                self.notes[k][0] += 1
            else:
                self.notes[k][1] += 1

    def size(self) -> int:
        return len(self.notes)


def build_prior(ids, max_order):
    tables = {L: defaultdict(Counter) for L in range(1, max_order + 1)}
    for i in range(1, len(ids)):
        tok = ids[i]
        for L in range(1, max_order + 1):
            if i - L < 0:
                break
            tables[L][tuple(ids[i - L:i])][tok] += 1
    return tables


def backoff(ctx, tables, max_order):
    for L in range(min(max_order, len(ctx)), 0, -1):
        tab = tables[L].get(tuple(ctx[-L:]))
        if tab:
            return tab, sum(tab.values())
    return None, 0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["5m", "stack"], default="5m")
    ap.add_argument("--stack-mb", type=int, default=20)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=500, help="positions before notebook activates")
    ap.add_argument("--max-entries", type=int, default=200_000)
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
    print(f"[{tag}] V={V} model={len(model_ids):,} prior={len(prior_ids):,} "
          f"test={len(test_ids):,}", flush=True)

    print("Building prior + smoothed model...", flush=True)
    t0 = time.time()
    priors = build_prior(prior_ids, args.max_order)
    sm = build_smoothed(model_ids, V, order=3)
    lam = (0.2, 0.3, 0.5)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(42)
    n = len(test_ids) - args.max_order - 1
    idx = rng.choice(n, size=min(args.n_test, n), replace=False)

    # one pass over test; notebook accumulates online (recency signal)
    nb = NotebookPrior(max_entries=args.max_entries)
    stats = {"nb0": [0, 0], "nb1": [0, 0], "nb2": [0, 0]}
    nll = {"nb0": 0.0, "nb1": 0.0, "nb2": 0.0}
    acc = {"nb0": 0, "nb1": 0, "nb2": 0}
    covered = 0
    t0 = time.time()
    for pos, i in enumerate(idx):
        ctx = tuple(test_ids[max(0, i - args.max_order):i])
        target = test_ids[i]
        tab, tot = backoff(ctx, priors, args.max_order)
        lp = model_logp_all(sm, ctx, V, lam)
        target_in_prior = bool(tab and tab.get(target, 0) > 0)

        # NB-0: fixed beta
        b0 = args.beta
        # NB-1: notebook trust
        t = nb.trust(ctx)
        b1 = min(0.95, max(0.05, t))
        # NB-2: hybrid — notebook + static (match_len)
        if tab:
            mLen = 0
            for L in range(min(args.max_order, len(ctx)), 0, -1):
                if priors[L].get(tuple(ctx[-L:])):
                    mLen = L
                    break
            b2 = min(0.95, max(0.05, 0.5 * t + 0.5 * (mLen / args.max_order)))
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

        # after knowing the target: record verdict (recency signal)
        if pos >= args.warmup:
            nb.update(ctx, target_in_prior)
            if tab:
                covered += 1

    n = len(idx)
    print(f"\n=== NoteBook Prior [{tag}] n={n} ({time.time()-t0:.0f}s) ===")
    for key in ("nb0", "nb1", "nb2"):
        ppl = float(np.exp(nll[key] / n))
        print(f"  {key}: PPL={ppl:7.3f}  acc={acc[key]/n:.3f}")
    print(f"  notebook entries={nb.size():,} covered={covered/n:.3f}")

    res = {"tag": tag, "n": n, "notebook_entries": nb.size(),
           "ppl": {k: float(np.exp(nll[k] / n)) for k in nll},
           "acc": {k: acc[k] / n for k in acc}}
    json.dump(res, open(os.path.join(HERE, f"notebook_{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
