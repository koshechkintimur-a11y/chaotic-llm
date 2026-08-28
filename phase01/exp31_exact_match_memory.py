"""exp31_exact_match_memory.py — cheap associative memory (1 hash lookup).

THE "а что если": make the memory itself cheap, so no gate is needed.

Current memory = V-dim KN distribution (512 lookups, 132 µs/token) → gate
dilemma (exp28-30). Proposed: EXACT MATCH retrieval — 1 hash lookup → the
top candidate token for the current context. Boost it in the mixer's
logits: logits[w*] += γ·log(c). No distribution, no smoothing, ~1 µs.

Question: how much of KN's PPL improvement does exact-match retain, at
1/100 the cost? This decides whether the β-Architecture can run with
unconditional cheap memory (no gate, no conditional execution).

Compare on test (12K positions, same setup as exp30):
  mixer-only           PPL 32.78  (0 µs memory)
  KN β=0.9 (baseline)  PPL 10.92  (132 µs/token)
  exact-match + boost  PPL ?      (1 µs/token)

Variants: constant γ boost; γ·log(c) boost; top-k boost. γ tuned on val.
"""
import os
import sys
import time
import json
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp31_exact_match_memory")
os.makedirs(OUT, exist_ok=True)

W = 256
ORDER = 3
BETA = 0.9


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


# ---- BPE (must match exp30's tokenizer) ----
def make_bpe(text, vocab_size=512):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), 2_000_000)
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(load_chars(os.path.join(HERE, "corpus_test.txt"))).ids

# ---- exact-match memory: ctx(2) -> {w: count} ----
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1
print(f"exact-match memory: {len(ctx_counts):,} contexts")

# ---- load exp30 precomputed logits (val/test) ----
d = np.load(os.path.join(HERE, "exp30_utility_gate", "results", "exp30_extracted.npz"))
logits_va, logits_te = d["logits_va"], d["logits_te"]
y_va, y_te = d["y_va"], d["y_te"]
va_starts, te_starts = d["va_starts"], d["te_starts"]
lpmix_va = torch.log_softmax(torch.tensor(logits_va), -1).double().numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()
print(f"loaded: val {len(y_va)}, test {len(y_te)}")


def topk_candidate(starts, seq, K):
    """For each position: top-K candidates (w, log_count) via 1 context lookup."""
    N = len(starts)
    cand_w = np.full((N, K), -1, dtype=np.int64)
    cand_l = np.zeros((N, K), dtype=np.float64)
    for k, i in enumerate(starts):
        tpos = i + W
        ctx = tuple(seq[tpos - ORDER + 1:tpos])
        e = ctx_counts.get(ctx)
        if e:
            top = sorted(e.items(), key=lambda x: -x[1])[:K]
            for j, (w, c) in enumerate(top):
                cand_w[k, j] = w
                cand_l[k, j] = np.log(c)
    return cand_w, cand_l


K = 3
candw_va, candl_va = topk_candidate(va_starts, train_ids, K)
candw_te, candl_te = topk_candidate(te_starts, test_ids, K)
hit_va = (candw_va[:, 0] != -1)
hit_te = (candw_te[:, 0] != -1)
print(f"memory hit rate: val {hit_va.mean()*100:.1f}%  test {hit_te.mean()*100:.1f}%")


def eval_boost(lpmix, ys, starts, seq, cand_w, cand_l, mode, gamma, K=3):
    """Boost mixer logits with exact-match candidates. Returns PPL, top1."""
    N = len(ys)
    nll = np.zeros(N)
    acc = 0
    for k in range(N):
        logits = lpmix[k].copy()
        for j in range(K):
            w = cand_w[k, j]
            if w < 0:
                break
            c = cand_l[k, j]
            if mode == "const":
                boost = gamma
            elif mode == "logc":
                boost = gamma * c
            elif mode == "logc_scaled":
                boost = gamma * c  # c = log(count)
            logits[w] += boost
        fused = logits - logsumexp(logits)
        nll[k] = -fused[ys[k]]
        acc += int(fused.argmax() == ys[k])
    return float(np.exp(np.mean(nll))), acc / N


def logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


# ---- tune γ on val, apply to test ----
print("\n=== tuning γ on val (mode=logc, top-1) ===")
best = None
for gamma in [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]:
    p, a = eval_boost(lpmix_va, y_va, va_starts, train_ids, candw_va[:, :1], candl_va[:, :1], "logc", gamma, 1)
    print(f"  γ={gamma:.1f}: PPL {p:.3f} top1 {a*100:.1f}%")
    if best is None or p < best[0]:
        best = (p, gamma)

_, g_best = best
print(f"best γ (val): {g_best}")

print("\n=== test, top-1 exact match + logc boost ===")
for gamma in [g_best, 1.0, 2.0]:
    p, a = eval_boost(lpmix_te, y_te, te_starts, test_ids, candw_te[:, :1], candl_te[:, :1], "logc", gamma, 1)
    print(f"  γ={gamma:.1f}: PPL {p:.3f} top1 {a*100:.1f}%")

print("\n=== test, top-3 exact match + logc boost ===")
for gamma in [0.5, 1.0, 2.0]:
    p, a = eval_boost(lpmix_te, y_te, te_starts, test_ids, candw_te, candl_te, "logc", gamma, 3)
    print(f"  γ={gamma:.1f}: PPL {p:.3f} top1 {a*100:.1f}%")

print("\n=== test, const boost (γ on top-1) ===")
for gamma in [0.5, 1.0, 2.0, 4.0]:
    p, a = eval_boost(lpmix_te, y_te, te_starts, test_ids, candw_te[:, :1], candl_te[:, :1], "const", gamma, 1)
    print(f"  γ={gamma:.1f}: PPL {p:.3f} top1 {a*100:.1f}%")

# ---- wall-clock: exact-match memory vs KN ----
print("\n=== wall-clock (memory lookup per token) ===")
# KN cost (no memo)
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1

def kn_dist_nomemo(ctx):
    n = len(ctx) + 1
    c_h = cnt[n - 1].get(ctx, 0)
    if c_h == 0:
        return None
    p = np.zeros(V, dtype=np.float64)
    n1 = 0
    row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0:
            n1 += 1
            p[w] = max(c_hw - 0.75, 0) / c_h
    return p

sample_ctx = [tuple(test_ids[te_starts[k] + W - ORDER + 1:te_starts[k] + W]) for k in range(2000)]
t0 = time.perf_counter()
for c in sample_ctx:
    kn_dist_nomemo(c)
t1 = time.perf_counter()
kn_cost = (t1 - t0) / len(sample_ctx)

t0 = time.perf_counter()
for k in range(2000):
    _ = ctx_counts.get(sample_ctx[k])
t1 = time.perf_counter()
em_cost = (t1 - t0) / 2000

print(f"  KN full dist:   {kn_cost*1e6:.2f} µs/token")
print(f"  exact match:    {em_cost*1e6:.2f} µs/token")
print(f"  speedup:        {kn_cost/em_cost:.1f}×")

# ---- summary vs baselines ----
print("\n=== summary ===")
r = {}
r["mixer_only"] = float(np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in range(len(y_te))])))
print(f"  mixer-only:           PPL {r['mixer_only']:.3f}")

# KN baseline: proper KN (memoized, with backoff)
# Build memoized KN for test contexts
def kn_dist_memo(ctx):
    if ctx in memo:
        return memo[ctx]
    n = len(ctx) + 1
    c_h = cnt[n - 1].get(ctx, 0)
    if c_h == 0:
        res = kn_dist_memo(ctx[1:])
        memo[ctx] = res
        return res
    p = np.zeros(V, dtype=np.float64)
    n1 = 0
    row = cnt[n]
    for w in range(V):
        c_hw = row.get(ctx + (w,), 0)
        if c_hw > 0:
            n1 += 1
            p[w] = max(c_hw - 0.75, 0) / c_h
    p_lower = kn_dist_memo(ctx[1:])
    p += (0.75 / c_h) * n1 * p_lower
    p /= p.sum()
    memo[ctx] = p
    return p

memo = {(): np.full(V, 1.0/V)}
nll_kn = np.zeros(len(y_te))
for k in range(len(y_te)):
    tpos = te_starts[k] + W
    ctx = tuple(test_ids[tpos - ORDER + 1:tpos])
    lp_mem = np.log(np.maximum(kn_dist_memo(ctx), 1e-12))
    pm = lpmix_te[k, y_te[k]]
    nll_kn[k] = -np.logaddexp(np.log1p(-BETA) + pm, np.log(BETA) + lp_mem[y_te[k]])
r["kn_baseline"] = float(np.exp(np.mean(nll_kn)))
print(f"  KN β=0.9 (memo):      PPL {r['kn_baseline']:.3f}")

# sparse MLE prior: only observed continuations (~5 lookups), no backoff
def sparse_mle_ppl(beta=BETA):
    nll = np.zeros(len(y_te))
    for k in range(len(y_te)):
        tpos = te_starts[k] + W
        ctx = tuple(test_ids[tpos - ORDER + 1:tpos])
        e = ctx_counts.get(ctx)
        pm = lpmix_te[k, y_te[k]]
        if e and sum(e.values()) > 1:
            tot = sum(e.values())
            logp = np.full(V, -1e9, dtype=np.float64)
            for w, c in e.items():
                logp[w] = np.log(c / tot)
            fused = np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + logp[y_te[k]])
        else:
            fused = pm
        nll[k] = -fused
    return float(np.exp(np.mean(nll)))

for beta_s in [0.3, 0.5, 0.9]:
    p = sparse_mle_ppl(beta_s)
    print(f"  sparse MLE β={beta_s}:   PPL {p:.3f}")
    if beta_s == 0.5:
        r["sparse_mle"] = p

# wall-clock: sparse MLE (lookup + iterate continuations + build V-dim array)
t0 = time.perf_counter()
for k in range(2000):
    ctx = sample_ctx[k]
    e = ctx_counts.get(ctx)
    if e:
        tot = sum(e.values())
        for w, c in e.items():
            _ = np.log(c / tot)
t1 = time.perf_counter()
sparse_cost = (t1 - t0) / 2000
print(f"  sparse MLE cost: {sparse_cost*1e6:.2f} µs/token")
print(f"  vs KN: {kn_cost/sparse_cost:.0f}× cheaper")

# exact match (best): recompute at g_best
p_best, a_best = eval_boost(lpmix_te, y_te, te_starts, test_ids, candw_te[:, :1], candl_te[:, :1], "logc", g_best, 1)
r["exact_match_best_ppl"] = p_best
r["exact_match_best_top1"] = a_best
gain_kn = r["mixer_only"] - r["kn_baseline"]
gain_em = r["mixer_only"] - p_best
r["exact_match_gain_retained"] = gain_em / gain_kn
print(f"  exact-match (γ={g_best}): PPL {p_best:.3f} top1 {a_best*100:.1f}%")
print(f"  KN gain: {gain_kn:.2f} PPL, exact-match gain: {gain_em:.2f} PPL")
print(f"  exact-match: {r['exact_match_gain_retained']*100:.0f}% of KN's gain at {kn_cost/em_cost:.0f}× cheaper")

# sparse MLE gain
if "sparse_mle" in r:
    gain_sm = r["mixer_only"] - r["sparse_mle"]
    r["sparse_mle_gain_retained"] = gain_sm / gain_kn
    print(f"  sparse MLE: {r['sparse_mle_gain_retained']*100:.0f}% of KN's gain at {kn_cost/sparse_cost:.0f}× cheaper")

json.dump(r, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("\nsaved", OUT)
