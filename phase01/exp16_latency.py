"""exp16_latency.py — Step 5c: honest latency/FLOPs benchmark at W=512-2048.

Measures forward-pass wall-clock for:
  - Hierarchical Chaotic Mixer (local windows 64 + padded relay)
  - Full Attention (single head, d=64)
at W = 512, 1024, 2048. Auto-detects CUDA (RTX 3060) if torch CUDA build is
available; otherwise CPU with the same measurement, honestly labeled.

Also reports analytic FLOP counts per forward pass.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp16_latency")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
WIDTHS = [512, 1024, 2048]
BATCHES = [1, 8]


# ============ Hierarchical chaotic mixer (scales to any W) ============
class ChaoticBlock(nn.Module):
    def __init__(self, Wl, d, bl, bg, g2_pad=None):
        super().__init__()
        self.Wl, self.d = Wl, d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
                       for t in range(1, bl + 1)}
        # relay grid: dynamic per call

    def _chaotic(self, h, sigmas, gates, grid):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, h, sig_g, pad):
        B, W, d = h.shape
        Nw = W // self.Wl
        hw = h.view(B, Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l, self.Wl)
                           for wi in range(Nw)], dim=1)
        g = loc.mean(dim=2)  # (B, Nw, d)
        if pad > 0:
            g = torch.cat([g, torch.zeros(B, pad, d, device=h.device)], dim=1)
        g = self._chaotic(g, sig_g, self.gates_g, g.shape[1])
        g = g[:, :Nw]
        gvec = g.mean(dim=1, keepdim=True)
        return loc.reshape(B, W, d) + gvec


class HierChaotic(nn.Module):
    def __init__(self, W, d):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(512, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.block = ChaoticBlock(W_LOCAL, d, BLOCKS_LOCAL, BLOCKS_GLOBAL)
        Nw = W // W_LOCAL
        g = int(math.ceil(math.sqrt(Nw)))
        if (g * g) % 2 == 1:
            g += 1  # coupling needs an even grid
        self.g = g
        self.pad = g * g - Nw
        self._sig_g = {t: torch.as_tensor(permute_indices(g * g, t), dtype=torch.long)
                       for t in range(1, BLOCKS_GLOBAL + 1)}

    def forward(self, x):
        h = self.embed(x) + self.pos
        return self.block(h, self._sig_g, self.pad)


class FullAttn(nn.Module):
    def __init__(self, W, d):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(512, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, W = x.shape
        h = self.embed(x) + self.pos
        qkv = self.qkv(h).reshape(B, W, 3, self.d).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = attn.softmax(-1)
        return self.norm(h + self.proj(attn @ v))


def flops_chaotic(W, d=64, Wl=64, bl=8, bg=4):
    """Analytic FLOPs: local windows (permute+coupling) + relay."""
    Nw = W // Wl
    g = int(math.ceil(math.sqrt(Nw)))
    if (g * g) % 2 == 1:
        g += 1
    g2 = g * g
    local = Nw * bl * Wl * d * 3          # permute (index) + 2 couplings
    relay = bg * g2 * d * 3
    return int(local + relay + W * d * 2)  # embed + readout-ish


def flops_attn(W, d=64):
    return int(W * W * d * 3 + W * d * d * 2)


def bench(model, W, bs, device, iters=50, warmup=10):
    model = model.to(device)
    model.eval()
    x = torch.randint(0, 500, (bs, W), device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt * 1e3  # ms per forward


results = {"device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
           "d": D_MODEL, "w_local": W_LOCAL}
print(f"Device: {DEVICE}")

for W in WIDTHS:
    hc = HierChaotic(W, D_MODEL)
    fa = FullAttn(W, D_MODEL)
    print(f"\n--- W={W} ---")
    print(f"FLOPs chaotic: {flops_chaotic(W):,}  attention: {flops_attn(W):,}  "
          f"ratio: {flops_attn(W)/max(flops_chaotic(W),1):.0f}x")
    results[f"W{W}"] = {"flops_chaotic": flops_chaotic(W),
                        "flops_attn": flops_attn(W),
                        "ratio": flops_attn(W) / max(flops_chaotic(W), 1)}
    for bs in BATCHES:
        t_hc = bench(hc, W, bs, DEVICE)
        t_fa = bench(fa, W, bs, DEVICE)
        print(f"  batch={bs}: chaotic {t_hc:.2f} ms | attn {t_fa:.2f} ms")
        results[f"W{W}"][f"chaotic_ms_b{bs}"] = t_hc
        results[f"W{W}"][f"attn_ms_b{bs}"] = t_fa
        results[f"W{W}"][f"attn_over_chaotic_b{bs}"] = t_fa / max(t_hc, 1e-6)

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
