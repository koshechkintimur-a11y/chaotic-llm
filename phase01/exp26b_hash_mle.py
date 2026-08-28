"""exp26b_hash_mle.py — clean isolation: hash storage vs dict storage.

Q: is the probing hash table FAITHFUL to the dict? Same data, same MLE
distributions, same β — only the STORAGE differs.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices


class ProbeTable:
    """Linear-probing hash table (local copy — avoids running exp26's main)."""

    def __init__(self, n_slots, order, bits_per_tok):
        self.n = int(n_slots)
        self.order = order
        self.bpt = bits_per_tok
        self.keys = np.full(self.n, -1, dtype=np.int64)
        self.counts = np.zeros(self.n, dtype=np.int64)
        self.n_distinct = 0

    def _pack(self, ngram):
        k = 0
        for t in ngram:
            k = (k << self.bpt) | int(t)
        return k

    @staticmethod
    def _mix(k):
        k = (k ^ (k >> 33)) & 0xFFFFFFFFFFFFFFFF
        k = (k * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        return (k ^ (k >> 33))

    def insert(self, ngram, count=1):
        key = self._pack(ngram)
        s = self._mix(key) % self.n
        while self.keys[s] != -1 and self.keys[s] != key:
            s = (s + 1) % self.n
        if self.keys[s] == -1:
            if self.n_distinct >= self.n:
                return False
            self.keys[s] = key
            self.n_distinct += 1
        self.counts[s] += count
        return True

    def lookup(self, ngram):
        key = self._pack(ngram)
        s = self._mix(key) % self.n
        while self.keys[s] != -1:
            if self.keys[s] == key:
                return self.counts[s]
            s = (s + 1) % self.n
        return 0

    def mem_mb(self):
        return (self.n * 12) / 1e6

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp26_hash_table_vocab")
torch.manual_seed(0)
np.random.seed(0)

W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
MAX_TRAIN_BYTES = 2_000_000
N_EVAL = 15000


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


def make_bpe(text, vocab_size):
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


tok = make_bpe(train_text, 512)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids


# ============ model (exp22 V1) ============
class ChaoticBlock(nn.Module):
    def __init__(self, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
                       for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, bg + 1)}

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
        B, W, d = h.shape
        hw = h.view(B, self.Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        gvec = glob.mean(dim=1, keepdim=True)
        return loc.reshape(B, W, d) + gvec


class ChaoticBase(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.block = ChaoticBlock(W, Wl, d, bl, bg)
        self.norm = nn.LayerNorm(d)

    def mix(self, x):
        h = self.embed(x) + self.pos
        return self.norm(h + self.block(h))


class ModelV1(nn.Module):
    def __init__(self, base, V):
        super().__init__()
        self.base = base
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(),
                                     nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1(ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V)
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                                 weights_only=True))
model.eval()


# ============ build order-3 counts in BOTH dict and hash ============
ORDER = 3
print("building order-3 counts...")
dict_cnt = defaultdict(int)
for i in range(ORDER, len(train_ids)):
    dict_cnt[tuple(train_ids[i - ORDER:i])] += 1
print(f"dict distinct order-3: {len(dict_cnt):,}")

# hash table (slot_factor 3)
slots = int(len(dict_cnt) * 3) + 10
tab = ProbeTable(slots, ORDER, 9)
for g, c in dict_cnt.items():
    tab.insert(g, c)
print(f"hash distinct: {tab.n_distinct:,}, mem {tab.mem_mb():.2f} MB")


def logp_from_dist(dist, ctx, V=V):
    """MLE with backoff from a per-context distribution dict. ctx = window."""
    for back in range(ORDER - 1, 0, -1):
        c = tuple(ctx[-back:])
        d = dist.get(c)
        if d:
            tot = sum(d.values())
            out = np.full(V, -1e9, dtype=np.float64)
            for t, c_ in d.items():
                out[t] = np.log(c_ / tot)
            return out
    return np.full(V, -1e9, dtype=np.float64)

# Precompute per-context MLE distributions (dict) — same as exp22
dict_dist = defaultdict(dict)
for g, c in dict_cnt.items():
    ctx, w = g[:ORDER - 1], g[ORDER - 1]
    dict_dist[ctx][w] = dict_dist[ctx].get(w, 0) + c

# hash-based per-context distributions: probe c(ctx,w) for all w
hash_dist = {}
hash_ctx_total = {}
for ctx in dict_dist:
    dist = {}
    tot = 0
    for w in range(V):
        c = tab.lookup(ctx + (w,))
        if c > 0:
            dist[w] = c
            tot += c
    hash_dist[ctx] = dist
    hash_ctx_total[ctx] = tot
print(f"hash contexts: {len(hash_dist):,}")


def evaluate(get_logp, beta, n_samples=N_EVAL):
    model.eval()
    nll_p, acc_p = [], 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            logits = model(torch.tensor([ctx], dtype=torch.long))
            logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
            cp = get_logp(ctx)
            if cp is not None:
                logp_c = np.logaddexp(np.log1p(-beta) + logp, np.log(beta) + cp)
            else:
                logp_c = logp
            nll_p.append(-logp_c[y])
            if int(np.argmax(logp_c)) == y:
                acc_p += 1
            if len(nll_p) >= n_samples:
                break
    n = len(nll_p)
    model.train()
    return {"ppl": float(np.exp(np.mean(nll_p))), "acc": acc_p / n}


beta = 0.3
r_dict = evaluate(lambda ctx: logp_from_dist(dict_dist, ctx), beta)
r_hash = evaluate(lambda ctx: logp_from_dist(hash_dist, ctx), beta)
print(f"\nβ={beta}: dict-MLE  PPL {r_dict['ppl']:.2f} top1 {r_dict['acc']*100:.1f}%")
print(f"       hash-MLE  PPL {r_hash['ppl']:.2f} top1 {r_hash['acc']*100:.1f}%")

# sanity: hash vs dict storage agreement
mismatch = sum(1 for ctx in dict_dist if dict_dist[ctx] != hash_dist.get(ctx, {}))
print(f"contexts with dict/hash disagreement: {mismatch}/{len(dict_dist)}")

with open(os.path.join(OUT, "results_hash_mle.json"), "w") as f:
    json.dump({"dict_mle": r_dict, "hash_mle": r_hash,
               "disagreements": mismatch, "hash_mem_MB": tab.mem_mb()}, f, indent=2)
print("saved")
