"""exp27_generation.py — Phase 5, exp27: end-to-end TEXT GENERATION.

The β-Architecture as a real generative model: autoregressive sampling with
the chaotic mixer (compute) + KN n-gram table (memory) + β-gate.

Seeds: real code prompts. Samples at several β to SHOW the gate working:
  β=0.0  — pure compute (mixer only)
  β=0.3  — mixed
  β=0.9  — memory-dominant
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp27_generation")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
MAX_TRAIN_BYTES = 2_000_000
ORDER = 3
GEN_LEN = 150
TEMPERATURE = 0.8
TOPP = 0.9


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


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


print("Training BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids


# ============ KN table (dict, order-3) ============
print("building KN order-3 table...")
cnt = [defaultdict(int) for _ in range(ORDER + 1)]
for i in range(1, len(train_ids)):
    for n in range(1, ORDER + 1):
        if i - n >= 0:
            cnt[n][tuple(train_ids[i - n:i])] += 1
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


# ============ model ============
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
print(f"model loaded ({sum(p.numel() for p in model.parameters()):,} params)")


# ============ sampling ============
def sample(logp, temp=TEMPERATURE, topp=TOPP):
    p = np.exp(logp - logp.max())
    p = p ** (1 / temp)
    p /= p.sum()
    # top-p
    idx = np.argsort(-p)
    cum = np.cumsum(p[idx])
    keep = idx[cum <= topp]
    if len(keep) == 0:
        keep = idx[:1]
    p[~np.isin(np.arange(V), keep)] = 0
    p /= p.sum()
    return int(np.random.choice(V, p=p))


def generate(seed_text, beta, length=GEN_LEN):
    ids = tok.encode(seed_text).ids
    if len(ids) > W:
        ids = ids[-W:]
    out_text = seed_text
    for _ in range(length):
        ctx = ids[-W:] if len(ids) >= W else ids
        pad = W - len(ctx)
        x = torch.tensor([[-1] * pad + ctx], dtype=torch.long)  # placeholder
        # handle pad: mask out embedding for -1
        x_padded = torch.tensor([ctx], dtype=torch.long)
        if len(ctx) < W:
            # build with repeated first token to fill window (positions matter via pos)
            fill = ctx[0] if ctx else 0
            full = [fill] * (W - len(ctx)) + ctx
            x_padded = torch.tensor([full], dtype=torch.long)
        with torch.no_grad():
            logits = model(x_padded)
        logp_m = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
        p_kn = kn_dist(tuple(ids[-ORDER + 1:])) if len(ids) >= ORDER - 1 else P_UNI
        logp_kn = np.log(p_kn + 1e-12)
        fused = np.logaddexp(np.log1p(-beta) + logp_m, np.log(beta) + logp_kn)
        nxt = sample(fused)
        ids.append(nxt)
        out_text += tok.decode([nxt])
    return out_text


seeds = [
    ("code_fn", "def fibonacci(n):"),
    ("code_cls", "class User:\n    def __init__(self, name):"),
    ("code_api", "app.get('/users', async (req, res) => {"),
]

results = {}
for sname, seed in seeds:
    print(f"\n===== {sname} =====\nseed: {seed}")
    results[sname] = {"seed": seed, "samples": {}}
    for beta in [0.0, 0.3, 0.9]:
        torch.manual_seed(0)
        np.random.seed(0)
        gen = generate(seed, beta)
        print(f"\n--- β={beta} ---\n{gen}\n")
        results[sname]["samples"][str(beta)] = gen

import json
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved to", OUT)
