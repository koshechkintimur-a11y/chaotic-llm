"""noise_test.py — run the noise retrieval test separately.

Main results + sweep already saved. This script only runs the noise test
(train on synthetic retrieval task, compare attention/chaotic/eye).
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from chaotic_gears import ChaoticBlock
from eye import SelectiveChaoticLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
V = 512
D = 128
EYE_K = 42


def best_window(W):
    """Largest perfect-square divisor of W that is ≤ W//4 (Arnold needs squares)."""
    best = 1
    for wl in range(1, W // 4 + 1):
        if W % wl == 0:
            r = int(wl ** 0.5)
            if r * r == wl:
                best = wl
    return best


def build_attention(W=256, d=D):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
            mask = torch.triu(torch.full((W, W), float("-inf")), diagonal=1)
            self.register_buffer("mask", mask)
        def forward(self, x):
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self.mask)
            x = x + a
            return x + self.ffn(self.ln2(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.pos = nn.Parameter(torch.randn(1, W, d)*0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(6)])
            self.ln_f = nn.LayerNorm(d)
            self.head = nn.Linear(d, V)
        def forward(self, x):
            h = self.embed(x) + self.pos
            for b in self.blocks:
                h = b(h)
            return self.head(self.ln_f(h[:, -1, :]))
    return Model()


def build_chaotic(W=256, d=D):
    wl = best_window(W)
    block = ChaoticBlock(W, wl, d, 8, 4)
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.pos = nn.Parameter(torch.randn(1, W, d)*0.02)
            self.block = block
            self.norm = nn.LayerNorm(d)
        def mix(self, x):
            return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = Base()
            self.readout = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d, V))
        def forward(self, x):
            h = self.base.mix(x)
            return self.readout(torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1))
    return Model()


def build_eye(W=256, d=D):
    wl = best_window(W)
    block = ChaoticBlock(W, wl, d, 8, 4)
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.pos = nn.Parameter(torch.randn(1, W, d)*0.02)
            self.block = block
            self.norm = nn.LayerNorm(d)
        def mix(self, x):
            return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))
    return SelectiveChaoticLM(Base(), V, d, "C", "soft", 1.0, EYE_K)


def make_noise_batch(n, W, noise_frac=0.95):
    """Task: noise tokens are small values {1..10}, KEY is a large value
    [100..511] at a random position. Model must output the KEY's value.
    Content-addressing: find the standout token, copy its value."""
    X = np.zeros((n, W), dtype=np.int64)
    Y = np.zeros(n, dtype=np.int64)
    for b in range(n):
        key_val = np.random.randint(100, 512)          # the standout KEY
        kp = np.random.randint(0, W)                    # random position
        X[b] = np.random.randint(1, 11, size=W)         # noise from small set
        X[b, kp] = key_val
        Y[b] = key_val
    return X, Y


def train_noise_model(model, W, steps=4000):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for s in range(steps):
        Xb, Yb = make_noise_batch(64, W)
        X = torch.tensor(Xb, dtype=torch.long, device=DEVICE)
        Y = torch.tensor(Yb, dtype=torch.long, device=DEVICE)
        opt.zero_grad()
        if isinstance(model, SelectiveChaoticLM):
            logits, _ = model(X)
        else:
            logits = model(X)
        loss = lossf(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    Xe, Ye = make_noise_batch(1000, W)
    with torch.no_grad():
        if isinstance(model, SelectiveChaoticLM):
            logits, _ = model(torch.tensor(Xe, dtype=torch.long, device=DEVICE))
        else:
            logits = model(torch.tensor(Xe, dtype=torch.long, device=DEVICE))
    return (logits.argmax(-1) == torch.tensor(Ye, device=DEVICE)).float().mean().item()


print("=== NOISE RETRIEVAL TEST (one KEY among small-value noise) ===")
noise = {}
for name, builder in [("attention", build_attention),
                      ("chaotic", build_chaotic),
                      ("chaotic+eye", build_eye)]:
    print(f"\n{name}:")
    noise[name] = {}
    for W in [64, 128, 256, 512]:
        m = builder(W=W)
        acc = train_noise_model(m, W, steps=3000)
        noise[name][W] = round(acc * 100, 1)
        print(f"  W={W} (noise {(W-1)/W:.1%}): acc {noise[name][W]}%", flush=True)

json.dump(noise, open(os.path.join(OUT, "noise_test.json"), "w"), indent=2)
print("saved", os.path.join(OUT, "noise_test.json"))