"""exp32_sparse_mle_nl.py — does the sparse-MLE pivot generalize to NL?

exp31 (code): sparse MLE retained 93% of KN's PPL gain at 17× lower cost.
Question: does this hold on natural language (WikiText-2)?

The NL n-gram distribution is sparser than code (more variety, fewer exact
repeats) — the honest risk is that no-backoff sparse MLE collapses where
KN's smoothing matters.

Setup mirrors exp23/31: NL mixer (nl_mixer.pt), BPE-512, order-3, windows
[i, i+W) → seq[i+W]. Compare mixer-only / mixer+KN / mixer+sparse-MLE.
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
OUT = os.path.join(HERE, "exp32_sparse_mle_nl")
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


print("BPE (NL)...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}")

# KN counts
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1
print(f"NL contexts: {len(ctx_counts):,}")

cont = defaultdict(int)
for (x, w), c in cnt[2].items():
    if c > 0:
        cont[w] += 1
total_cont = sum(cont.values())
P_UNI = np.zeros(V, dtype=np.float64)
for w in range(V):
    cw = cont.get(w, 0)
    P_UNI[w] = max(cw - 0.75, 0) / total_cont
P_UNI += (0.75 * len(cont)) / total_cont / V
P_UNI /= P_UNI.sum()
memo = {(): P_UNI}


def kn_dist(ctx):
    if ctx in memo:
        return memo[ctx]
    n = len(ctx) + 1
    c_h = cnt[n - 1].get(ctx, 0)
    if c_h == 0:
        res = kn_dist(ctx[1:])
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
    p_lower = kn_dist(ctx[1:])
    p += (0.75 / c_h) * n1 * p_lower
    p /= p.sum()
    memo[ctx] = p
    return p


def kn_logp(ctx):
    return np.log(np.maximum(kn_dist(ctx), 1e-12))


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
print("NL mixer loaded")

# positions
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
    feats = np.zeros((N, D_MODEL * 2), dtype=np.float32)
    logits_all = np.zeros((N, V), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, BATCH):
            e = min(s + BATCH, N)
            h = model.base.mix(torch.tensor(X[s:e], dtype=torch.long))
            gvec = h.mean(dim=1)
            feat = torch.cat([h[:, -1, :], gvec], dim=-1)
            logits_all[s:e] = model.readout(feat).numpy()
            feats[s:e] = feat.numpy()
    return feats, logits_all, ys


print("extract val/test...")
_, logits_va, y_va = extract(va_starts, train_ids)
_, logits_te, y_te = extract(te_starts, test_ids)
lpmix_va = torch.log_softmax(torch.tensor(logits_va), -1).double().numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()


def eval_memory(starts, seq, lpmix, ys, kind, beta):
    """kind: 'kn' (full, memo) or 'sparse' (observed continuations only)."""
    N = len(ys)
    nll = np.zeros(N)
    acc = 0
    for k in range(N):
        tpos = starts[k] + W
        ctx = tuple(seq[tpos - ORDER + 1:tpos])
        pm = lpmix[k, ys[k]]
        if kind == "kn":
            lp_mem = kn_logp(ctx)
        else:
            e = ctx_counts.get(ctx)
            if e and sum(e.values()) > 0:
                tot = sum(e.values())
                logp = np.full(V, -1e9, dtype=np.float64)
                for w, c in e.items():
                    logp[w] = np.log(c / tot)
                lp_mem = logp
            else:
                lp_mem = np.full(V, -1e9, dtype=np.float64)
        fused = np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem[ys[k]])
        nll[k] = -fused
    return float(np.exp(np.mean(nll)))


# tune β on val for both
print("\n=== β tuning (val) ===")
best = {}
for kind in ["kn", "sparse"]:
    for beta in [0.3, 0.5, 0.7, 0.9]:
        p = eval_memory(va_starts, train_ids, lpmix_va, y_va, kind, beta)
        print(f"  {kind} β={beta}: PPL {p:.3f}")
        if kind not in best or p < best[kind][0]:
            best[kind] = (p, beta)

print("\n=== test ===")
mixer_only = float(np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in range(len(y_te))])))
print(f"  mixer-only:        PPL {mixer_only:.3f}")
res = {"mixer_only": mixer_only}
for kind in ["kn", "sparse"]:
    beta_b = best[kind][1]
    p = eval_memory(te_starts, test_ids, lpmix_te, y_te, kind, beta_b)
    res[kind] = {"ppl": p, "beta": beta_b}
    print(f"  mixer+{kind} β={beta_b}: PPL {p:.3f}")

gain_kn = res["mixer_only"] - res["kn"]["ppl"]
gain_sp = res["mixer_only"] - res["sparse"]["ppl"]
res["sparse_retained_fraction"] = gain_sp / gain_kn
print(f"  KN gain: {gain_kn:.2f}, sparse gain: {gain_sp:.2f}")
print(f"  sparse retains {res['sparse_retained_fraction']*100:.0f}% of KN's gain")

# wall-clock on NL
print("\n=== wall-clock (NL) ===")
sample_ctx = [tuple(test_ids[te_starts[k] + W - ORDER + 1:te_starts[k] + W]) for k in range(2000)]

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

t0 = time.perf_counter()
for c in sample_ctx:
    kn_dist_nomemo(c)
t1 = time.perf_counter()
kn_cost = (t1 - t0) / len(sample_ctx)

t0 = time.perf_counter()
for k in range(2000):
    e = ctx_counts.get(sample_ctx[k])
    if e:
        tot = sum(e.values())
        for w, c in e.items():
            _ = np.log(c / tot)
t1 = time.perf_counter()
sp_cost = (t1 - t0) / 2000
res["kn_cost_us"] = kn_cost * 1e6
res["sparse_cost_us"] = sp_cost * 1e6
res["speedup"] = kn_cost / sp_cost
print(f"  KN: {kn_cost*1e6:.2f} µs/token | sparse: {sp_cost*1e6:.2f} µs/token | {kn_cost/sp_cost:.0f}×")

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("\nsaved", OUT)
