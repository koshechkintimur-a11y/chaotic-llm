"""exp10_hierarchical_lm.py — Phase 0.1, Experiment 10 (Step 1).

CLOSE THE LONG-RANGE GAP: hierarchical chaotic mixing.

The known weakness (exp05): a single chaotic window cannot handle
long-range (info diluted through the coupling chain).  Fix: hierarchy.
  - context = 256 chars = 4 windows of 64
  - Level 1 (local): 8 chaotic blocks on each 8x8 window
  - Level 2 (global): pool windows to 4 representations, chaotic-mix on a
    2x2 grid, broadcast the global vector back to every window
  - readout at the last position (local + global info)

Cost per step: 4*64*8 + 4*4 ~ 2K ops  vs  attention 256^2 = 65K  (~30x).

Probe (is long-range actually used?):
  - PPL with context 64 (local only)  vs  256 (hierarchical).
  - If hierarchical PPL < local PPL, far context is used -> long-range works.

Compared against AttnLM (window 256) on the same real-code corpus.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp10_hierarchical_lm")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

W = 256          # full context
W_LOCAL = 64     # local window (8x8)
N_WINDOWS = W // W_LOCAL  # 4
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
MAX_TRAIN_CHARS = 1_500_000
BATCH = 128
EPOCHS = 1


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return list(text[:limit]) if limit else list(text)


train_chars = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_CHARS)
test_chars = load_chars(os.path.join(HERE, "corpus_test.txt"))
cnt = Counter(train_chars)
vocab = [c for c, _ in cnt.most_common(160)]
char2id = {c: i for i, c in enumerate(vocab)}
V = len(vocab)
print(f"vocab {V}, train {len(train_chars):,}, test {len(test_chars):,}")


def encode(chars):
    return [char2id.get(c, 0) for c in chars]


# ============ Hierarchical ChaoticLM ============
class HierChaoticLM(nn.Module):
    """Local chaotic windows + global chaotic relay over window reps."""
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
                       for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, bg + 1)}
        self.readout = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, V))

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, x, use_global=True):
        B, W = x.shape
        d = self.d
        h = self.embed(x) + self.pos[:, :W, :]
        if W == self.Wl:
            # local-only path (probe): one window, no global relay
            loc = self._chaotic(h, self._sig_l, self.gates_l)   # (B, 64, d)
            last = loc[:, -1, :]
            gvec = torch.zeros(B, d, device=h.device)
            return self.readout(torch.cat([last, gvec], dim=-1))
        hw = h.view(B, self.Nw, self.Wl, d)               # (B, 4, 64, d)
        loc = []
        for wi in range(self.Nw):
            loc.append(self._chaotic(hw[:, wi], self._sig_l, self.gates_l))
        loc = torch.stack(loc, dim=1)                    # (B, 4, 64, d)
        glob = loc.mean(dim=2)                           # (B, 4, d)
        glob = self._chaotic(glob, self._sig_g, self.gates_g)  # (B, 4, d)
        gvec = glob.mean(dim=1, keepdim=True)            # (B, 1, d)
        last = loc[:, -1, -1, :]                         # (B, d)
        return self.readout(torch.cat([last, gvec.squeeze(1)], dim=-1))


class AttnLM(nn.Module):
    def __init__(self, V, W, d):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, x, use_global=True):
        B, W = x.shape
        h = self.embed(x) + self.pos
        qkv = self.qkv(h).reshape(B, W, 3, self.d).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = attn.softmax(-1)
        h = self.norm(h + self.proj(attn @ v))
        return self.readout(h[:, -1, :])


# ============ training ============
ids = np.array(encode(train_chars), dtype=np.int64)

def make_batches(W, bs):
    n = len(ids) - W - 1
    idx = np.random.permutation(n)[: (n // bs) * bs].reshape(-1, bs)
    for bi in idx:
        X = np.stack([ids[i:i + W] for i in bi])
        Y = ids[bi + W]
        yield torch.tensor(X), torch.tensor(Y)

def train_model(model, W, lr=1e-3, epochs=EPOCHS, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    steps = 0
    t0 = time.time()
    for ep in range(epochs):
        total, n = 0.0, 0
        for X, Y in make_batches(W, BATCH):
            opt.zero_grad()
            loss = lossf(model(X), Y)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1; steps += 1
            if steps % 2000 == 0:
                print(f"  [{tag} {steps:,} steps] loss={total/n:.3f} ({time.time()-t0:.0f}s)")

def evaluate(model, W, ctx_len, n_samples=30000):
    """PPL/top-1 with a context of `ctx_len` chars (last ctx_len)."""
    model.eval()
    nll, acc = [], 0
    use_g = ctx_len >= 2 * W_LOCAL  # global relay only when full context given
    with torch.no_grad():
        for i in range(0, min(len(test_ids) - W - 1, n_samples * 32), 32):
            c = ctx_len
            ctx_ids = test_ids[i + W - c:i + W]
            y = int(test_ids[i + W])
            logits = model(torch.tensor(ctx_ids[None, :]), use_global=use_g)
            logp = torch.log_softmax(logits[0], -1).numpy()
            nll.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc += 1
            if len(nll) >= n_samples:
                break
    n = len(nll)
    model.train()
    return {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n}

results = {"W": W, "W_local": W_LOCAL, "blocks_local": BLOCKS_LOCAL,
           "blocks_global": BLOCKS_GLOBAL, "d": D_MODEL, "vocab": V}

test_ids = np.array(encode(test_chars), dtype=np.int64)

print("\n=== Hierarchical ChaoticLM ===")
hcm = HierChaoticLM(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
print(f"params: {sum(p.numel() for p in hcm.parameters()):,}")
train_model(hcm, W, tag="hier")
r_64 = evaluate(hcm, W, 64)
r_256 = evaluate(hcm, W, W)
print(f"HierChaoticLM ctx=64 : {r_64}")
print(f"HierChaoticLM ctx=256: {r_256}")
results["hierarchical"] = {"ctx64": r_64, "ctx256": r_256,
                           "longrange_gain": r_64["ppl"] / r_256["ppl"]}

print("\n=== AttnLM (window 256) ===")
am = AttnLM(V, W, D_MODEL)
print(f"params: {sum(p.numel() for p in am.parameters()):,}")
train_model(am, W, tag="attn")
r_a256 = evaluate(am, W, W)
print(f"AttnLM ctx=256: {r_a256}")
results["attention"] = {"ctx256": r_a256}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
