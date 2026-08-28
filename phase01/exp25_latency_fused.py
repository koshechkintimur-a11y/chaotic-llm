"""exp25_latency_fused.py — Phase 5, exp25: close the wall-clock gap.

exp16 showed the chaotic mixer is 76-225× SLOWER than attention on GPU —
but the dominant cost was the Python loop over WINDOWS (8 blocks × Nw windows
= 260 sequential iterations at W=2048, each launching kernels).

Fix: the local permutation σ_t is IDENTICAL for every window, so batch all
windows into one tensor (B·Nw, 64, d) and run the 8 blocks ONCE — 12 total
iterations instead of 8·Nw+4.

Benchmark (RTX 3060): eager-per-window vs vectorized vs vectorized+compile
(no triton on Windows — report honestly) vs attention (cuBLAS).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp25_latency_fused")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D = 64
WIDTHS = [512, 1024, 2048, 4096]
BATCHES = [1, 8]


class ChaoticBlockEager(nn.Module):
    """Original: Python loop over windows (the exp16 bottleneck)."""

    def __init__(self, Wl, d, bl, bg):
        super().__init__()
        self.Wl, self.d = Wl, d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, h, sig_l, sig_g, pad):
        B, W, d = h.shape
        Nw = W // self.Wl
        hw = h.view(B, Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], sig_l, self.gates_l)
                           for wi in range(Nw)], dim=1)
        g = loc.mean(dim=2)
        if pad > 0:
            g = torch.cat([g, torch.zeros(B, pad, d, device=h.device)], dim=1)
        g = self._chaotic(g, sig_g, self.gates_g)
        g = g[:, :Nw]
        gvec = g.mean(dim=1, keepdim=True)
        return loc.reshape(B, W, d) + gvec


class ChaoticBlockVec(nn.Module):
    """Vectorized: batch ALL windows into one tensor, run 8 blocks once."""

    def __init__(self, Wl, d, bl, bg):
        super().__init__()
        self.Wl, self.d = Wl, d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))

    def forward(self, h, sig_l, sig_g, pad):
        B, W, d = h.shape
        Nw = W // self.Wl
        # local: batch all windows (B*Nw, Wl, d); permutation identical per window
        hl = h.view(B * Nw, self.Wl, d)
        for t in range(1, len(self.gates_l) + 1):
            hl = hl[:, sig_l[t].to(h.device), :]
            g = torch.sigmoid(self.gates_l[t - 1])
            even = hl[:, 0::2, :]
            odd = hl[:, 1::2, :]
            hl = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B * Nw, self.Wl, d)
        loc = hl.view(B, Nw, self.Wl, d)
        # relay on window means
        g = loc.mean(dim=2)                      # (B, Nw, d)
        if pad > 0:
            g = torch.cat([g, torch.zeros(B, pad, d, device=h.device)], dim=1)
        for t in range(1, len(self.gates_g) + 1):
            g = g[:, sig_g[t].to(h.device), :]
            gg = torch.sigmoid(self.gates_g[t - 1])
            even = g[:, 0::2, :]
            odd = g[:, 1::2, :]
            g = torch.stack([even + gg * odd, odd + gg * even], dim=2).reshape(B, g.shape[1], d)
        g = g[:, :Nw]
        gvec = g.mean(dim=1, keepdim=True)
        return loc.reshape(B, W, d) + gvec


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


def make_sigmas(W, Wl, bl, bg):
    Nw = W // Wl
    g = int(math.ceil(math.sqrt(Nw)))
    if (g * g) % 2 == 1:
        g += 1
    g2 = g * g
    pad = g2 - Nw
    sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
             for t in range(1, bl + 1)}
    sig_g = {t: torch.as_tensor(permute_indices(g2, t), dtype=torch.long)
             for t in range(1, bg + 1)}
    return sig_l, sig_g, pad


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def flops_chaotic(W, d=64, Wl=64, bl=8, bg=4):
    Nw = W // Wl
    g = int(math.ceil(math.sqrt(Nw)))
    if (g * g) % 2 == 1:
        g += 1
    g2 = g * g
    local = Nw * bl * Wl * d * 3
    relay = bg * g2 * d * 3
    return int(local + relay)


def flops_attn(W, d=64):
    return int(W * W * d * 3 + W * d * d * 2)


results = {"device": str(DEVICE), "d": D, "w_local": W_LOCAL,
           "triton": False, "torch": torch.__version__}
print(f"Device: {DEVICE}")

for W in WIDTHS:
    sig_l, sig_g, pad = make_sigmas(W, W_LOCAL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
    blk_e = ChaoticBlockEager(W_LOCAL, D, BLOCKS_LOCAL, BLOCKS_GLOBAL).to(DEVICE)
    blk_v = ChaoticBlockVec(W_LOCAL, D, BLOCKS_LOCAL, BLOCKS_GLOBAL).to(DEVICE)
    # torch.compile attempt (Windows, no triton — expect eager fallback)
    try:
        blk_c = torch.compile(ChaoticBlockVec(W_LOCAL, D, BLOCKS_LOCAL, BLOCKS_GLOBAL).to(DEVICE))
        compiled_ok = True
    except Exception as e:
        compiled_ok = False
        print(f"  compile failed: {type(e).__name__}")
    fa = FullAttn(W, D).to(DEVICE)

    x = torch.randint(0, 500, (8, W), device=DEVICE)
    h_e = torch.randn(8, W, D, device=DEVICE)
    h_v = torch.randn(8, W, D, device=DEVICE)

    print(f"\n--- W={W} (FLOPs: chaos {flops_chaotic(W):,} vs attn {flops_attn(W):,}) ---")
    results[f"W{W}"] = {"flops_chaos": flops_chaotic(W), "flops_attn": flops_attn(W)}
    for bs in BATCHES:
        xe = x[:bs]
        he = h_e[:bs]
        hv = h_v[:bs]
        t_attn = bench(lambda: fa(xe))
        t_eager = bench(lambda: blk_e(he, sig_l, sig_g, pad))
        t_vec = bench(lambda: blk_v(hv, sig_l, sig_g, pad))
        t_comp = None
        if compiled_ok:
            t_comp = bench(lambda: blk_c(hv, sig_l, sig_g, pad))
        print(f"  bs={bs}: attn {t_attn:7.3f}ms | eager {t_eager:7.3f}ms "
              f"| vec {t_vec:7.3f}ms" + (f" | compile {t_comp:7.3f}ms" if t_comp else ""))
        results[f"W{W}"][f"attn_b{bs}"] = t_attn
        results[f"W{W}"][f"eager_b{bs}"] = t_eager
        results[f"W{W}"][f"vec_b{bs}"] = t_vec
        if t_comp:
            results[f"W{W}"][f"compile_b{bs}"] = t_comp

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
