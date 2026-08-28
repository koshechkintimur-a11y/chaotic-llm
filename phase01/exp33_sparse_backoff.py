"""exp33_sparse_backoff.py — cheap backoff closes the NL gap.

exp32 showed sparse MLE retains 87% of KN's gain on NL, but the absolute
gap is bigger on NL (28.3 vs 23.2 PPL) than code (11.66 vs 10.94) because
NL contexts are sparser and KN's lower-order backoff matters.

"А что если": hierarchical sparse backoff — if the order-3 context is rare
(c_h small), fall back to the order-2 sparse MLE, then order-1. Cost: 1
lookup for strong contexts, 2-3 for rare ones — still ~10× cheaper than KN.

Design: sparse MLE over observed continuations, with interpolation
λ(c_h) blending the current order toward the next lower order:
  P(w|h) = (1-λ)·MLE_n(w|h) + λ·P_{n-1}(w|h')
λ = α/(α + c_h)  (more evidence → less backoff), α tuned on val.
All distributions stay sparse (only observed continuations get mass).
"""
import os
import sys
import time
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp33_sparse_backoff")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
ORDER = 3
MAX_TRAIN_CHARS = 4_000_000

N_TR, N_VA, N_TE = 60_000, 10_000, 12_000
BATCH = 1024


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_train.txt"), MAX_TRAIN_CHARS)
test_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_test.txt"))


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


print("BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids

# per-order context -> {w: count} (order 2..3)
ctx_n = {}
for n in range(2, ORDER + 1):
    d = defaultdict(dict)
    for i in range(n - 1, len(train_ids)):
        ctx = tuple(train_ids[i - (n - 1):i])
        w = train_ids[i]
        dd = d[ctx]
        dd[w] = dd.get(w, 0) + 1
    ctx_n[n] = d
# unigram counts
uni = defaultdict(int)
for t in train_ids:
    uni[t] += 1
tot_uni = sum(uni.values())


def sparse_logp(ctx, n):
    """log-probs over observed continuations of ctx at order n (ctx has n-1 tokens).
    Returns (logp_array, total_evidence)."""
    d = ctx_n[n].get(ctx)
    if not d:
        return None, 0
    tot = sum(d.values())
    logp = np.full(V, -1e9, dtype=np.float64)
    for w, c in d.items():
        logp[w] = np.log(c / tot)
    return logp, tot


def backoff_logp(ctx_tokens, alpha):
    """Hierarchical sparse MLE with λ=α/(α+c_h) interpolation toward lower order."""
    # ctx_tokens = last ORDER-1 tokens (length ORDER-1)
    logp = np.full(V, -1e9, dtype=np.float64)
    # order-3
    p3, c3 = sparse_logp(tuple(ctx_tokens), 3)
    if p3 is None:
        # fall to order-2
        p2, c2 = sparse_logp(tuple(ctx_tokens[1:]), 2)
        if p2 is None:
            # fall to unigram
            for w, c in uni.items():
                logp[w] = np.log(c / tot_uni)
            return logp
        return p2
    lam = alpha / (alpha + c3)
    p2, c2 = sparse_logp(tuple(ctx_tokens[1:]), 2)
    if p2 is None:
        # blend order-3 with unigram
        for w, c in uni.items():
            logp[w] = np.log(c / tot_uni)
        lp3 = np.where(p3 > -1e8, p3 + np.log1p(-lam), -1e9)
        lpu = np.where(logp > -1e8, logp + np.log(lam), -1e9)
        return np.logaddexp(lp3, lpu)
    # blend order-3 with order-2 (both sparse)
    lp3 = np.where(p3 > -1e8, p3 + np.log1p(-lam), -1e9)
    lp2 = np.where(p2 > -1e8, p2 + np.log(lam), -1e9)
    return np.logaddexp(lp3, lp2)


# model
class ChaoticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nw = W // W_LOCAL
        self.gates_l = nn.Parameter(torch.zeros(BLOCKS_LOCAL))
        self.gates_g = nn.Parameter(torch.zeros(BLOCKS_GLOBAL))
        self._sig_l = {t: torch.as_tensor(permute_indices(W_LOCAL, t), dtype=torch.long) for t in range(1, BLOCKS_LOCAL + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long) for t in range(1, BLOCKS_GLOBAL + 1)}

    def _chaotic(self, h, sigmas, gates):
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even, odd = h[:, 0::2, :], h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(h.shape[0], h.shape[1], D_MODEL)
        return h

    def forward(self, h):
        B, Wd, d = h.shape
        hw = h.view(B, self.Nw, W_LOCAL, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l) for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        return loc.reshape(B, Wd, d) + glob.mean(dim=1, keepdim=True)


class ChaoticBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, W, D_MODEL) * 0.02)
        self.block = ChaoticBlock()
        self.norm = nn.LayerNorm(D_MODEL)

    def mix(self, x):
        return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))


class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ChaoticBase()
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1()
model.load_state_dict(torch.load(os.path.join(HERE, "exp23_nl_beta", "nl_mixer.pt"), weights_only=True))
model.eval()

rng = np.random.default_rng(42)
maxstart_tr = len(train_ids) - W - 1
maxstart_te = len(test_ids) - W - 1
all_tr = np.sort(rng.choice(maxstart_tr, size=N_TR + N_VA, replace=False))
tr_starts, va_starts = all_tr[:N_TR], all_tr[N_TR:]
te_starts = np.sort(rng.choice(maxstart_te, size=N_TE, replace=False))


def extract(starts, seq):
    N = len(starts)
    X = np.zeros((N, W), dtype=np.int64)
    ys = np.zeros(N, dtype=np.int64)
    for k, i in enumerate(starts):
        X[k] = seq[i:i + W]
        ys[k] = seq[i + W]
    logits_all = np.zeros((N, V), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, BATCH):
            e = min(s + BATCH, N)
            h = model.base.mix(torch.tensor(X[s:e], dtype=torch.long))
            gvec = h.mean(dim=1)
            logits_all[s:e] = model.readout(torch.cat([h[:, -1, :], gvec], dim=-1)).numpy()
    return logits_all, ys


print("extract...")
logits_va, y_va = extract(va_starts, train_ids)
logits_te, y_te = extract(te_starts, test_ids)
lpmix_va = torch.log_softmax(torch.tensor(logits_va), -1).double().numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()


def eval_backoff(starts, seq, lpmix, ys, alpha, beta):
    N = len(ys)
    nll = np.zeros(N)
    for k in range(N):
        tpos = starts[k] + W
        ctx = tuple(seq[tpos - ORDER + 1:tpos])
        pm = lpmix[k, ys[k]]
        lp_mem = backoff_logp(ctx, alpha)
        fused = np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem[ys[k]])
        nll[k] = -fused
    return float(np.exp(np.mean(nll)))


# tune α, β on val
print("\n=== tune α, β (val) ===")
best = (1e9, None)
for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
    for beta in [0.5, 0.7, 0.9]:
        p = eval_backoff(va_starts, train_ids, lpmix_va, y_va, alpha, beta)
        if p < best[0]:
            best = (p, (alpha, beta))
        print(f"  α={alpha} β={beta}: PPL {p:.3f}")

print(f"best: α={best[1][0]} β={best[1][1]} PPL {best[0]:.3f}")
alpha_b, beta_b = best[1]

print("\n=== test ===")
p_sp = eval_backoff(te_starts, test_ids, lpmix_te, y_te, alpha_b, beta_b)
print(f"  sparse+backoff (α={alpha_b}, β={beta_b}): PPL {p_sp:.3f}")

# wall-clock
print("\n=== wall-clock ===")
sample = te_starts[:2000]
t0 = time.perf_counter()
for k in range(2000):
    tpos = sample[k] + W
    _ = backoff_logp(tuple(test_ids[tpos - ORDER + 1:tpos]), alpha_b)
t1 = time.perf_counter()
cost = (t1 - t0) / 2000
print(f"  sparse+backoff cost: {cost*1e6:.2f} µs/token")

json.dump({"best": best, "test_ppl": p_sp, "cost_us": cost * 1e6},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)
