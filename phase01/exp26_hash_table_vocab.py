"""exp26_hash_table_vocab.py — Phase 5, exp26: memory channel at scale.

Q: does the memory-scaling advantage survive (a) hash-based storage
(probing table, like KenLM) and (b) a 4× larger vocabulary (BPE-2048)?

Part A (fast, vocab 512): dict-KN vs hash-KN — isolate the hash effect.
Part B (retrain, vocab 2048): tiny mixer + hash-KN order-3 — vocab scaling.
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp26_hash_table_vocab")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
BATCH = 64
MAX_TRAIN_BYTES = 2_000_000
N_EVAL = 15000
D_KN = 0.75


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


# ============ probing hash table (KenLM-style) ============
class ProbeTable:
    """Linear-probing hash table: pack(tokens) -> count."""

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
                return False  # table full — entry dropped (memory budget)
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
        return (self.n * 12) / 1e6  # int64 key + int64 count


# ============ Kneser-Ney over hash tables (order N, backoff to dict lower orders) ============
class HashKN:
    """Interpolated KN using hash tables at each order (lower orders exact dicts)."""

    def __init__(self, train_ids, order, bits_per_tok, slot_factor=3.0):
        self.V = 1 << bits_per_tok
        self.order = order
        self.bpt = bits_per_tok
        self.tables = {}
        # build hash tables for orders 2..order
        for n in range(2, order + 1):
            # count distinct n-grams first (for slot sizing)
            dist = {}
            for i in range(n, len(train_ids)):
                g = tuple(train_ids[i - n:i])
                dist[g] = dist.get(g, 0) + 1
            slots = int(len(dist) * slot_factor) + 10
            tab = ProbeTable(slots, n, bits_per_tok)
            for g, c in dist.items():
                tab.insert(g, c)
            self.tables[n] = (tab, len(dist))
        # unigram continuation counts (exact, small)
        self.cont = defaultdict(int)
        for i in range(2, len(train_ids)):
            self.cont[train_ids[i]] += 1  # simplified: use frequency unigram
        self._memo = {}
        # precompute lower-order KN distributions lazily via dict fallback tables
        self._lower = [None] * (order + 1)  # placeholder

    def c(self, n, g):
        return self.tables[n][0].lookup(g)

    def kn_dist(self, ctx):
        """P_KN over V for context tuple (length order-1, backoffs)."""
        if ctx in self._memo:
            return self._memo[ctx]
        n = len(ctx) + 1
        if n == 1:
            tot = sum(self.cont.values()) or 1
            p = np.zeros(self.V, dtype=np.float64)
            for w, c in self.cont.items():
                p[w] = max(c - D_KN, 0) / tot
            p /= p.sum()
            self._memo[ctx] = p
            return p
        # count of context at order n-1
        c_h = 0
        # c(h) = sum of continuations; we compute N1 and the max-prob terms by probing
        row = self.tables.get(n)
        p = np.zeros(self.V, dtype=np.float64)
        n1 = 0
        if row is not None:
            tab = row[0]
            # iterate continuations via probing all w (V lookups) — acceptable
            base = tab._pack(ctx)
            # probe c(ctx+w) for each w
            for w in range(self.V):
                c_hw = tab.lookup(ctx + (w,))
                if c_hw > 0:
                    n1 += 1
                    c_h += c_hw
                    p[w] = max(c_hw - D_KN, 0)
        if c_h == 0:
            res = self.kn_dist(ctx[1:])
            self._memo[ctx] = res
            return res
        p /= c_h
        p_lower = self.kn_dist(ctx[1:])
        p += (D_KN / c_h) * n1 * p_lower
        p /= p.sum()
        self._memo[ctx] = p
        return p


# ============ model classes (shared) ============
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


def evaluate(model, get_logp, beta, test_ids, n_samples=N_EVAL):
    model.eval()
    nll_m, nll_p, acc_m, acc_p = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            logits = model(torch.tensor([ctx], dtype=torch.long))
            logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
            nll_m.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc_m += 1
            cp = get_logp(test_ids, i + W)
            if cp is not None:
                logp_c = np.logaddexp(np.log1p(-beta) + logp, np.log(beta) + cp)
            else:
                logp_c = logp
            nll_p.append(-logp_c[y])
            if int(np.argmax(logp_c)) == y:
                acc_p += 1
            if len(nll_m) >= n_samples:
                break
    n = len(nll_m)
    model.train()
    return {"n": n, "ppl_model": float(np.exp(np.mean(nll_m))),
            "ppl_fused": float(np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_fused": acc_p / n}


results = {}

# ============ PART A: hash vs dict at vocab 512 (reuse exp22 mixer) ============
print("=== PART A: vocab 512, hash vs dict ===")
tok512 = make_bpe(train_text, 512)
V512 = tok512.get_vocab_size()
train512 = tok512.encode(train_text).ids
test512 = tok512.encode(test_text).ids
model512 = ModelV1(ChaoticBase(V512, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V512)
model512.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                                    weights_only=True))
model512.eval()

# build hash-KN order-3 at vocab 512
hkn = HashKN(train512, 3, bits_per_tok=9)
memo = {}
def get_hash_logp(ids, i):
    ctx = tuple(ids[i - 2:i])
    if ctx not in memo:
        p = hkn.kn_dist(ctx)
        memo[ctx] = np.log(p + 1e-12)
    return memo[ctx]
rA = evaluate(model512, get_hash_logp, 0.5, test512)
memA = sum(t[0].mem_mb() for t in hkn.tables.values())
print(f"hash-KN order-3 (vocab512): PPL_fused={rA['ppl_fused']:.2f} "
      f"top1={rA['acc_fused']*100:.1f}% mem≈{memA:.1f}MB")
results["partA_hash_kn_v512"] = {"ppl": rA["ppl_fused"], "acc": rA["acc_fused"],
                                 "mem_MB": memA}

# ============ PART B: vocab 2048, retrain mixer + hash-KN ============
print("\n=== PART B: vocab 2048 ===")
tok2 = make_bpe(train_text, 2048)
V2 = tok2.get_vocab_size()
train2 = tok2.encode(train_text).ids
test2 = tok2.encode(test_text).ids
print(f"BPE-2048: {len(train2):,} train tokens, {len(test2):,} test tokens")

model2 = ModelV1(ChaoticBase(V2, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V2)
ckpt2 = os.path.join(OUT, "mixer_v2048.pt")
if os.path.exists(ckpt2):
    model2.load_state_dict(torch.load(ckpt2, weights_only=True))
    print("loaded mixer_v2048")
else:
    print(f"training mixer at vocab {V2} ({sum(p.numel() for p in model2.parameters()):,} params)")
    opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    n = len(train2) - W - 1
    for s in range(6000):
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train2[i:i + W] for i in bi]), dtype=torch.long)
        Y = torch.tensor([train2[i + W] for i in bi], dtype=torch.long)
        opt.zero_grad()
        loss = lossf(model2(X), Y)
        loss.backward()
        opt.step()
        if s % 3000 == 0:
            print(f"  [mixer2 {s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    torch.save(model2.state_dict(), ckpt2)
    print("mixer_v2048 trained")

# hash-KN order-3 at vocab 2048
print("building hash-KN order-3 at vocab 2048...")
hkn2 = HashKN(train2, 3, bits_per_tok=11)
memo2 = {}
def get_hash_logp2(ids, i):
    ctx = tuple(ids[i - 2:i])
    if ctx not in memo2:
        p = hkn2.kn_dist(ctx)
        memo2[ctx] = np.log(p + 1e-12)
    return memo2[ctx]

# mixer alone
rB0 = evaluate(model2, lambda ids, i: None, 0.0, test2)
print(f"mixer alone (v2048): PPL {rB0['ppl_model']:.2f}, top-1 {rB0['acc_model']*100:.1f}%")
# fused
rB = evaluate(model2, get_hash_logp2, 0.5, test2)
memB = sum(t[0].mem_mb() for t in hkn2.tables.values())
print(f"hash-KN order-3 (v2048): PPL_fused={rB['ppl_fused']:.2f} "
      f"top1={rB['acc_fused']*100:.1f}% mem≈{memB:.1f}MB")
results["partB"] = {"vocab": V2, "mixer_alone_ppl": rB0["ppl_model"],
                    "mixer_alone_acc": rB0["acc_model"],
                    "fused_ppl": rB["ppl_fused"], "fused_acc": rB["acc_fused"],
                    "mem_MB": memB}
# reference: vocab 512 dict-KN order-3 fused at beta .5
results["reference_v512_dictkn"] = {"ppl": 13.79, "acc": 0.407, "mem_MB": 3.54}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
