"""analyze_memory.py — what does MEM-B actually store? (analysis, no training)

Reports for MEM-B (content-addressable):
  1. Pattern length distribution in the hash table (how many patterns per L=8..16)
  2. Miss rate on test (pattern not found) — coverage vs capacity
  3. Top-20 most frequent patterns (do they look meaningful, not random?)

Also dumps comparable stats for MEM-0 (order-3) and MEM-A (order-8) for context.

Usage:  python analyze_memory.py [subsample_tokens]
"""
import os
import sys
import json
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, REPO)

from experiment import load_chars, build_tokenizer, VOCAB, MAX_TRAIN, W, ORDER
from memory import (BaseMemory, Order8Memory, ContentAddressableMemory,
                    HybridMemory, RandomMemory)

N_EVAL = 5000


def get_ids(limit):
    tok = build_tokenizer()
    tr = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    te = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    return tok.encode(tr).ids[:limit], tok.encode(te).ids


def analyze(config_name, mem_factory, train_ids, test_ids, ctx_len):
    mem = mem_factory()
    mem.build(train_ids)
    print(f"\n=== {config_name} ===")
    print(f"  entries: {mem.n_entries():,}")
    print(f"  bytes  : {mem.size_bytes():,}")
    print(f"  ctx_len: {ctx_len}")

    # (2) miss rate on test (sample)
    rng = __import__("numpy").random.default_rng(7)
    maxstart = len(test_ids) - W - 1
    starts = np.sort(rng.choice(maxstart, size=N_EVAL, replace=False))
    miss = 0
    for i in starts:
        ctx = tuple(test_ids[i+W-ctx_len:i+W])
        if mem.query(ctx) is None:
            miss += 1
    miss_rate = miss / N_EVAL
    print(f"  miss_rate(test): {miss_rate:.3f}")

    # (1) + (3) for MEM-B only (has _patterns)
    if hasattr(mem, "_patterns") and mem._patterns:
        # length distribution
        L_counts = Counter(L for (L, _, _, _) in mem._patterns)
        print(f"  pattern length distribution (L: count):")
        for L in sorted(L_counts):
            print(f"    L={L:2d}: {L_counts[L]:,}")
        # top-20
        top = mem.top_patterns(20)
        print(f"  top-20 patterns (L | ctx_tokens | next_dist | total):")
        for (L, ctx, c, tot) in top:
            if ctx is None:
                ctx_str = "?"
            else:
                ctx_str = ",".join(str(t) for t in ctx)
            # next dist: top-3
            nxt = c.most_common(3)
            nxt_str = " ".join(f"{t}:{n}({n/tot:.2f})" for t, n in nxt)
            print(f"    L={L} | [{ctx_str}] | {nxt_str} | total={tot}")

    return {
        "config": config_name,
        "entries": mem.n_entries(),
        "bytes": mem.size_bytes(),
        "ctx_len": ctx_len,
        "miss_rate": miss_rate,
        "length_dist": dict(Counter(L for (L, _, _, _) in mem._patterns).items()) if hasattr(mem, "_patterns") else {},
    }


if __name__ == "__main__":
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("subsample", nargs="?", type=int, default=200_000)
    args = ap.parse_args()

    train_ids, test_ids = get_ids(args.subsample)
    print(f"train subsample: {len(train_ids):,}  test: {len(test_ids):,}")

    results = {}
    # MEM-0 (order-3)
    results["MEM0"] = analyze(
        "MEM0 (order-3)", lambda: BaseMemory(order=3, min_count=2),
        train_ids, test_ids, ctx_len=3)
    # MEM-A (order-8)
    results["MEM_A"] = analyze(
        "MEM_A (order-8)", lambda: Order8Memory(order=8, min_count=3),
        train_ids, test_ids, ctx_len=8)
    # MEM-B (content-addressable 8..16)
    results["MEM_B"] = analyze(
        "MEM_B (content-addr 8..16)",
        lambda: ContentAddressableMemory(min_len=8, max_len=16, min_count=2),
        train_ids, test_ids, ctx_len=16)

    json.dump(results, open(os.path.join(HERE, "analyze_memory.json"), "w"), indent=2)
    print("\nSaved analyze_memory.json")
