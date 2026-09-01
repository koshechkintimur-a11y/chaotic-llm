"""eval_prior_stand.py — CPU bench of n-gram prior (morin) on our code corpus.

Measures the prior ALONE (no neural model needed):
  - coverage (fraction of test contexts with any observation)
  - top-1 accuracy of the prior's argmax
  - PPL with add-k smoothing (prior-only model)
  - recall@k (target in top-k of prior)

This gives the honest ceiling for each prior variant BEFORE any mixture.

Usage: python eval_prior_stand.py [order] [subsample]
"""
import os
import sys
import json
import argparse
import time
from collections import defaultdict, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.dirname(PHASE))

from exp_memory_selector.experiment import load_chars, build_tokenizer, MAX_TRAIN, VOCAB

ORDER_DEFAULT = 3


def get_ids(limit=None):
    tok = build_tokenizer()
    tr = load_chars(os.path.join(PHASE, "corpus5m_train.txt"), MAX_TRAIN)
    te = load_chars(os.path.join(PHASE, "corpus5m_test.txt"))
    train_ids = tok.encode(tr).ids
    test_ids = tok.encode(te).ids
    if limit:
        train_ids = train_ids[:limit]
    return tok, train_ids, test_ids


def build_prior(train_ids, order):
    """prior[ctx_len][ctx_tuple] -> Counter(next_tok)"""
    tables = {k: defaultdict(Counter) for k in range(1, order + 1)}
    for i in range(1, len(train_ids)):
        tok = train_ids[i]
        for L in range(1, order + 1):
            if i - L < 0:
                break
            tables[L][tuple(train_ids[i - L:i])][tok] += 1
    return tables


def eval_prior(tables, order, test_ids, V, n_test=20000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n = len(test_ids) - order - 1
    idx = rng.choice(n, size=min(n_test, n), replace=False)
    cover = 0
    acc1 = 0
    recall5 = 0
    recall10 = 0
    nll = 0.0
    nll_cnt = 0
    total = len(idx)
    for i in idx:
        # find longest matching context
        best = None
        best_tot = 0
        for L in range(min(order, i), 0, -1):
            ctx = tuple(test_ids[i - L:i])
            tab = tables[L].get(ctx)
            if tab:
                tot = sum(tab.values())
                if tot > best_tot:
                    best, best_tot = tab, tot
                break  # longest match found (like morin backoff)
        target = test_ids[i]
        if best is None:
            continue
        cover += 1
        # top-1
        top1 = best.most_common(1)[0][0]
        if top1 == target:
            acc1 += 1
        topk = [t for t, _ in best.most_common(10)]
        if target in topk[:5]:
            recall5 += 1
        if target in topk[:10]:
            recall10 += 1
        # add-k smoothed NLL (k=1)
        tot = best_tot
        for t, c in best.items():
            p = (c + 1) / (tot + V)
            if t == target:
                nll += -np.log(p)
                nll_cnt += 1
                break
        else:
            p = 1 / (tot + V)
            nll += -np.log(p)
            nll_cnt += 1
    return {
        "order": order,
        "n_test": total,
        "coverage": cover / total,
        "top1_acc": acc1 / max(1, cover),
        "recall5": recall5 / max(1, cover),
        "recall10": recall10 / max(1, cover),
        "ppl_addk1": float(np.exp(nll / max(1, nll_cnt))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orders", nargs="*", type=int, default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--subsample", type=int, default=1_000_000)
    args = ap.parse_args()

    tok, train_ids, test_ids = get_ids(args.subsample)
    V = tok.get_vocab_size()
    print(f"V={V} train_tokens={len(train_ids):,} test_tokens={len(test_ids):,}", flush=True)

    results = {}
    for order in args.orders:
        t0 = time.time()
        tables = build_prior(train_ids, order)
        r = eval_prior(tables, order, test_ids, V)
        r["build_s"] = round(time.time() - t0, 1)
        results[f"order{order}"] = r
        print(f"  order={order}: cov={r['coverage']:.3f} top1={r['top1_acc']:.3f} "
              f"r5={r['recall5']:.3f} ppl={r['ppl_addk1']:.2f} "
              f"build={r['build_s']}s", flush=True)

    json.dump(results, open(os.path.join(HERE, "prior_stand.json"), "w"), indent=2)
    print("saved prior_stand.json")


if __name__ == "__main__":
    main()
