"""exp30_utility_gate.py — Utility-Gated Memory (conditional computation).

MAIN QUESTION:
  Can a cheap signal predict, BEFORE the expensive memory lookup, whether
  that lookup will actually help the current token — so we can PHYSICALLY
  skip memory computation without losing quality?

NOT a β experiment. This is conditional execution of compute.

Pipeline:
  1. Ground-truth utility ΔL = L_mixer - L_mixer+KN for every token
     (computed offline; the gate NEVER sees ΔL at decision time).
  2. Cheap features available BEFORE the KN lookup.
  3. Three gate levels: 0=membership, 1=hand rules, 2=tiny learned LR.
  4. Skip-vs-ΔPPL curve, quality-budget table, real wall-clock,
     generalization, long-range effect K=1..16.
  5. Verdict: SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED.
"""
import os
import sys
import time
import json
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp30_utility_gate")
RES = os.path.join(OUT, "results")
DOC = os.path.join(OUT, "docs")
os.makedirs(RES, exist_ok=True)
os.makedirs(DOC, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
ORDER = 3
MAX_TRAIN_BYTES = 2_000_000
BETA = 0.9          # fusion weight when memory IS used
BATCH = 1024

N_TR, N_VA, N_TE = 60_000, 10_000, 12_000


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)],
                            trainer=trainer)
    return tok


print("BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(load_chars(os.path.join(HERE, "corpus_test.txt"))).ids
print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}")

# ============ KN counts ============
print("KN counts...")
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
ctx_counts = defaultdict(dict)   # ctx(2) -> {w: count}  (order-3)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1

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


# ============ model ============
class ChaoticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nw = W // W_LOCAL
        self.gates_l = nn.Parameter(torch.zeros(BLOCKS_LOCAL))
        self.gates_g = nn.Parameter(torch.zeros(BLOCKS_GLOBAL))
        self._sig_l = {t: torch.as_tensor(permute_indices(W_LOCAL, t), dtype=torch.long)
                       for t in range(1, BLOCKS_LOCAL + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, BLOCKS_GLOBAL + 1)}

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, h):
        B, Wd, d = h.shape
        hw = h.view(B, self.Nw, W_LOCAL, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
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
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(),
                                     nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1()
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                                 weights_only=True))
model.eval()


# ============ positions: train / val / test ============
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


print("extracting train...")
feats_tr, logits_tr, y_tr = extract(tr_starts, train_ids)
print("extracting val...")
feats_va, logits_va, y_va = extract(va_starts, train_ids)
print("extracting test...")
feats_te, logits_te, y_te = extract(te_starts, test_ids)


def logprobs(logits):
    return torch.log_softmax(torch.tensor(logits), -1).double().numpy()


lpmix_tr, lpmix_va, lpmix_te = logprobs(logits_tr), logprobs(logits_va), logprobs(logits_te)


# ============ cheap KN features + ground-truth ΔL ============
def kn_feats_and_loss(starts, seq, lpmix, ys):
    """Per-token: cheap KN features (BEFORE lookup cost) + true KN loss.
    Returns arrays length N."""
    N = len(starts)
    hit = np.zeros(N, dtype=bool)          # context in table?
    c_h = np.zeros(N, dtype=np.float32)    # context total count
    n1 = np.zeros(N, dtype=np.float32)     # distinct continuations
    top_prob = np.zeros(N, dtype=np.float32)  # max cont count / c_h
    lpmem = np.zeros(N, dtype=np.float64)  # KN logp of true y (ground truth)
    mix_loss = np.zeros(N, dtype=np.float64)
    for k, i in enumerate(starts):
        tpos = i + W
        ctx = tuple(seq[tpos - ORDER + 1:tpos])
        e = ctx_counts.get(ctx)
        if e:
            hit[k] = True
            tot = sum(e.values())
            c_h[k] = tot
            n1[k] = len(e)
            top_prob[k] = max(e.values()) / tot
        lpmem[k] = kn_logp(ctx)[ys[k]]
        mix_loss[k] = -lpmix[k, ys[k]]
    return hit, c_h, n1, top_prob, lpmem, mix_loss


print("ground-truth features + ΔL...")
(htr, chtr, n1tr, tptr, lpmem_tr, Lmix_tr) = kn_feats_and_loss(tr_starts, train_ids, lpmix_tr, y_tr)
(hva, chva, n1va, tpva, lpmem_va, Lmix_va) = kn_feats_and_loss(va_starts, train_ids, lpmix_va, y_va)
(hte, chte, n1te, tpte, lpmem_te, Lmix_te) = kn_feats_and_loss(te_starts, test_ids, lpmix_te, y_te)


def fused_loss(lpmix, ys, lpmem, beta=BETA):
    """-log((1-β)p_mix + β p_mem) for the true token."""
    L = np.zeros(len(ys))
    for k in range(len(ys)):
        pm = lpmix[k, ys[k]]
        pk = lpmem[k]
        L[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + pk)
    return L


Lfull_tr = fused_loss(lpmix_tr, y_tr, lpmem_tr)
Lfull_va = fused_loss(lpmix_va, y_va, lpmem_va)
Lfull_te = fused_loss(lpmix_te, y_te, lpmem_te)

# ground-truth utility
dL_tr = Lmix_tr - Lfull_tr   # >0: memory helped
dL_va = Lmix_va - Lfull_va
dL_te = Lmix_te - Lfull_te

print(f"baseline PPL (always-on β={BETA}): "
      f"tr {np.exp(Lfull_tr.mean()):.3f} va {np.exp(Lfull_va.mean()):.3f} "
      f"te {np.exp(Lfull_te.mean()):.3f}")
print(f"mixer-only PPL: te {np.exp(Lmix_te.mean()):.3f}")

# ============ save ground-truth dataset ============
print("saving ground-truth...")
with open(os.path.join(RES, "exp30_utility_distribution.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["split", "context", "y", "memory_hit", "c_h", "n1", "top_prob",
                 "mixer_loss", "full_memory_loss", "delta_loss"])
    for k in range(N_TE):
        tpos = te_starts[k] + W
        ctx = tuple(test_ids[tpos - ORDER + 1:tpos])
        wr.writerow(["test", str(ctx), int(y_te[k]), int(hte[k]), float(chte[k]),
                     float(n1te[k]), float(tpte[k]), f"{Lmix_te[k]:.6f}",
                     f"{Lfull_te[k]:.6f}", f"{dL_te[k]:.6f}"])
print("saved utility distribution CSV")
