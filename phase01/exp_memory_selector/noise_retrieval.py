"""noise_retrieval.py — решающий шум-тест для MEM (ТЗ раздел 5).

Честная постановка: берём из TRAIN реальные позиции, где токен = KEY
(KEY встречается в корпусе, память его знает). Вокруг KEY ставим ШУМ.
Вопрос: при контексте [...шум, KEY] предсказывает ли память KEY как next?

То есть: может ли память "вспомнить" токен, который ТОЛЬКО ЧТО был в
контексте (сразу после KEY), когда вокруг — шум (95-99.5%).

Для order-3: KEY должен быть в последних 3 токенах → если сразу после
KEY идёт шум, order-3 НЕ увидит KEY (локален). Для content-addressable
(перекрытие 8..16): если KEY входил в длинный паттерн, может найти.

Метрика: retrieval accuracy = доля случаев, где KEY в top-k предсказаний
памяти при контексте [...шум, KEY].

Usage:  python noise_retrieval.py [config] [subsample]
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, REPO)

from experiment import load_chars, build_tokenizer, VOCAB, MAX_TRAIN
from memory import (BaseMemory, Order8Memory, ContentAddressableMemory,
                    HybridMemory, RandomMemory)

NOISE_TOKENS = list(range(1, 101))          # шум: токены 1..100
KEY_CANDIDATES = list(range(150, 511))       # KEY: редкие токены (вне шума)


def get_train_ids(limit):
    tok = build_tokenizer()
    tr = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    return tok.encode(tr).ids[:limit]


def build_memory_for(config, train_ids):
    if config == "MEM0":
        return BaseMemory(order=3, min_count=2).build(train_ids)
    if config == "MEM_A":
        return Order8Memory(order=8, min_count=3).build(train_ids)
    if config == "MEM_B":
        return ContentAddressableMemory(min_len=8, max_len=16, min_count=2).build(train_ids)
    if config == "MEM_C":
        return HybridMemory(local_order=3, min_len=8, max_len=16, alpha=0.5).build(train_ids)
    raise ValueError(config)


def pick_keys(train_ids, rng, n_keys=5):
    counts = Counter(train_ids)
    cands = [t for t in KEY_CANDIDATES if counts.get(t, 0) >= 30]
    return [int(t) for t in rng.choice(cands, size=min(n_keys, len(cands)), replace=False)]


def predict_from_memory(mem, ctx, topk=10):
    ctx_len = mem.ctx_len
    if len(ctx) < ctx_len:
        return []
    qctx = tuple(int(t) for t in ctx[-ctx_len:])
    r = mem.query(qctx)
    if r is None:
        return []
    counts, tot = r
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [int(tok) for tok, _ in ranked[:topk]]


def run(config, subsample, W_list, noise_fracs, n_trials=300):
    train_ids = get_train_ids(subsample)
    mem = build_memory_for(config, train_ids)
    print(f"[{config}] entries={mem.n_entries():,} ctx_len={mem.ctx_len}", flush=True)

    rng = np.random.default_rng(123)
    keys = pick_keys(train_ids, rng)
    print(f"  KEYS = {keys}", flush=True)

    results = {}
    for W in W_list:
        for nf in noise_fracs:
            ctx_len = mem.ctx_len
            acc1 = acc5 = acc10 = 0
            tested = 0
            for _ in range(n_trials):
                key = int(rng.choice(keys))
                # ищем в train позицию, где train_ids[i] == key (реальный контекст)
                # берём последние ctx_len токенов ДО key как контекст
                candidates = [i for i in range(ctx_len, len(train_ids)) if train_ids[i] == key]
                if not candidates:
                    continue
                i = int(rng.choice(candidates))
                # контекст = train_ids[i-ctx_len : i] (реальный, до KEY)
                ctx = list(train_ids[i - ctx_len:i])
                # заменяем долю nf токенов (кроме последнего, который перед KEY)
                # на шум — имитация "KEY в шуме"
                n_repl = int((len(ctx) - 1) * nf)
                repl_idx = list(rng.choice(len(ctx) - 1, size=n_repl, replace=False))
                for ri in repl_idx:
                    ctx[ri] = int(rng.choice(NOISE_TOKENS))
                # предсказание памяти: что после ctx? (ожидаем KEY)
                preds = predict_from_memory(mem, ctx, topk=10)
                if not preds:
                    continue
                tested += 1
                if key in preds[:1]:
                    acc1 += 1
                if key in preds[:5]:
                    acc5 += 1
                if key in preds[:10]:
                    acc10 += 1
            denom = tested if tested > 0 else 1
            results[f"W{W}_noise{int(nf*1000)}"] = {
                "acc1": round(acc1 / denom, 4),
                "acc5": round(acc5 / denom, 4),
                "acc10": round(acc10 / denom, 4),
                "tested": tested,
            }
            print(f"  W={W} noise={nf}: acc1={acc1/denom:.3f} "
                  f"acc5={acc5/denom:.3f} acc10={acc10/denom:.3f} (n={tested})", flush=True)
    return {"config": config, "keys": keys, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["MEM0", "MEM_A", "MEM_B", "MEM_C"])
    ap.add_argument("subsample", nargs="?", type=int, default=200_000)
    args = ap.parse_args()

    W_list = [64, 128, 256, 512]
    noise_fracs = [0.95, 0.99, 0.995]
    res = run(args.config, args.subsample, W_list, noise_fracs, n_trials=300)
    out = f"noise_retrieval_{args.config}.json"
    json.dump(res, open(os.path.join(HERE, out), "w"), indent=2)
    print(f"Saved {out}")
