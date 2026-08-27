"""exp09_chaotic_lm.py — Phase 0.1, Experiment 9.

THE SYNTHESIS ON REAL LANGUAGE DATA:
  Chaotic proposer (cheap reversible mixing) + beta corpus-prior filter.

Setup (mirrors morin-filter W but with a chaotic proposer instead of a
16B LLM):
  - Corpus: the user's real TypeScript/Python code, split 4:1 BY FILE.
  - Tokenization: character-level (vocab ~150 chars of code).
  - ChaoticLM: sliding window (64 chars = 8x8 grid), T=8 chaotic blocks
    (cat-map permutation + bounded symmetric coupling), readout at the
    window's last position -> next-char logits.
  - AttnLM: same window, 1 causal attention layer (baseline).
  - n-gram prior (order 3) from the TRAIN split only.
  - beta-filter: P_mix = (1-beta)*P_model + beta*P_prior.

Questions:
  Q1. Does the beta-filter improve the chaotic LM? (synthesis: yes)
  Q2. ChaoticLM+prior vs AttnLM+prior: does chaos keep up as a proposer?
  Q3. Does chaos add anything over the n-gram prior alone?
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
from chaos_lib import permute_indices, square_grid_size

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp09_chaotic_lm")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

WINDOW = 64        # 8x8 grid (perfect square for the cat map)
BLOCKS = 8         # chaotic blocks (log2(64)+2 = 8)
D_MODEL = 64
ORDER = 3          # n-gram prior order
BETA = 0.3
MAX_TRAIN_CHARS = 4_000_000
BATCH = 256
EPOCHS = 2


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return list(text[:limit]) if limit else list(text)


# ============ vocab ============
train_chars = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_CHARS)
test_chars = load_chars(os.path.join(HERE, "corpus_test.txt"))
from collections import Counter
cnt = Counter(train_chars)
vocab = [c for c, _ in cnt.most_common(160)]
char2id = {c: i for i, c in enumerate(vocab)}
V = len(vocab)
print(f"vocab size: {V}, train chars: {len(train_chars):,}, test chars: {len(test_chars):,}")

def encode(chars):
    return [char2id.get(c, 0) for c in chars]  # 0 = UNK

# ============ n-gram prior (TRAIN only) ============
def build_prior(chars, order=ORDER):
    ids = encode(chars)
    prior = defaultdict(lambda: defaultdict(int))
    for i in range(order, len(ids)):
        ctx = tuple(ids[i - order:i])
        prior[ctx][ids[i]] += 1
    return {k: dict(v) for k, v in prior.items()}

print("building n-gram prior...")
prior = build_prior(train_chars)
print(f"prior contexts: {len(prior):,}")

def prior_logp(ctx_ids):
    """log P over vocab for the last ORDER ids of ctx_ids; None if unseen."""
    for back in range(ORDER, 0, -1):
        table = prior.get(tuple(ctx_ids[-back:]))
        if table:
            tot = sum(table.values())
            out = np.full(V, -1e9, dtype=np.float64)
            for tok, c in table.items():
                out[tok] = np.log(c / tot)
            return out
    return None

# ============ models ============
class ChaoticLM(nn.Module):
    def __init__(self, V, W, d, blocks):
        super().__init__()
        self.W = W
        self.d = d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.gates = nn.Parameter(torch.zeros(blocks))
        self._sigmas = {t: torch.as_tensor(permute_indices(W, t), dtype=torch.long)
                        for t in range(1, blocks + 1)}
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, x):
        B, W = x.shape
        h = self.embed(x) + self.pos
        for t in range(1, len(self.gates) + 1):
            sig = self._sigmas[t].to(h.device)
            h = h[:, sig, :]
            g = torch.sigmoid(self.gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, W, self.d)
        q = h[:, -1, :]  # read out at the last (most recent) position
        return self.readout(q)


class AttnLM(nn.Module):
    def __init__(self, V, W, d):
        super().__init__()
        self.W = W
        self.d = d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, x):
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
def make_batches(ids, W, bs):
    n = len(ids) - W - 1
    idx = np.random.permutation(n)[: (n // bs) * bs].reshape(-1, bs)
    for batch_idx in idx:
        X = np.stack([ids[i:i + W] for i in batch_idx])
        Y = ids[batch_idx + W]
        yield torch.tensor(X), torch.tensor(Y)

def train_model(model, lr=1e-3, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    steps = 0
    t0 = time.time()
    for ep in range(epochs):
        total, n = 0.0, 0
        for X, Y in make_batches(ids, WINDOW, BATCH):
            opt.zero_grad()
            logits = model(X)
            loss = lossf(logits, Y)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
            steps += 1
            if steps % 1000 == 0:
                print(f"  [{steps:,} steps] loss={total/n:.3f} ({time.time()-t0:.0f}s)")
    return model

# ============ evaluation ============
test_ids = np.array(encode(test_chars), dtype=np.int64)
def evaluate(model):
    """PPL & top-1: model alone, model+prior(beta)."""
    model.eval()
    nll_base, nll_mix = [], []
    acc_base = acc_mix = 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - WINDOW - 1, 64):
            ctx_ids = test_ids[i:i + WINDOW]
            y = int(test_ids[i + WINDOW])
            logits = model(torch.tensor(ctx_ids[None, :]))
            logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
            nll_base.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc_base += 1
            cp = prior_logp(list(ctx_ids))
            if cp is not None:
                logp_c = np.logaddexp(np.log1p(-BETA) + logp, np.log(BETA) + cp)
            else:
                logp_c = logp
            nll_mix.append(-logp_c[y])
            if int(np.argmax(logp_c)) == y:
                acc_mix += 1
    n = len(nll_base)
    ppl_b = float(np.exp(np.mean(nll_base)))
    ppl_m = float(np.exp(np.mean(nll_mix)))
    model.train()
    return {"n": n, "ppl_model": ppl_b, "ppl_mix": ppl_m, "ppl_ratio": ppl_b / ppl_m,
            "acc_model": acc_base / n, "acc_mix": acc_mix / n}

results = {"window": WINDOW, "blocks": BLOCKS, "d": D_MODEL, "order": ORDER,
           "beta": BETA, "vocab": V, "corpus_train_chars": len(train_chars),
           "corpus_test_chars": len(test_chars), "models": {}}

print("\n=== training ChaoticLM ===")
cm = ChaoticLM(V, WINDOW, D_MODEL, BLOCKS)
print(f"params: {sum(p.numel() for p in cm.parameters()):,}")
train_model(cm)
r_cm = evaluate(cm)
print(f"ChaoticLM: {r_cm}")
results["models"]["ChaoticLM"] = r_cm

print("\n=== training AttnLM ===")
am = AttnLM(V, WINDOW, D_MODEL)
print(f"params: {sum(p.numel() for p in am.parameters()):,}")
train_model(am)
r_am = evaluate(am)
print(f"AttnLM: {r_am}")
results["models"]["AttnLM"] = r_am

# n-gram LM alone (no neural model): PPL/acc of the prior by itself
print("\n=== n-gram prior alone ===")
nll_p = []
acc_p = 0
for i in range(0, len(test_ids) - ORDER - 1, 64):
    ctx = list(test_ids[i:i + ORDER])
    y = int(test_ids[i + ORDER])
    cp = prior_logp(ctx)
    if cp is not None:
        nll_p.append(-cp[y])
        if int(np.argmax(cp)) == y:
            acc_p += 1
n = max(len(nll_p), 1)
r_prior = {"n": n, "ppl_prior": float(np.exp(np.mean(nll_p))),
           "acc_prior": acc_p / n, "coverage": len(nll_p) / max(1, (len(test_ids) - ORDER - 1) // 64 + 1)}
print(f"n-gram prior alone: {r_prior}")
results["models"]["ngram_prior"] = r_prior

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
