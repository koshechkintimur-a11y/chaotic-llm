"""models.py — PRIOR-PROP architectures (ТЗ П1-П5).

DP         — Distributed Prior: prior-embedding moves through SAME permutations as tokens
DP-noprop  — prior embedding exists but does NOT propagate (no mixer_p)
DP-rand    — prior embedding frozen random
C-cap      — capacity control: same params as DP, no prior (single mixer, larger d)
PM         — Propagating Memory: prior accumulates residually through mixer layers
SP         — Swarm Prior: agents at different scales

All share the Arnold-cat-map permutations (П1), zero-init new gates (П3).
"""
import math
import torch
import torch.nn as nn
import numpy as np

# import the chaotic building blocks
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
for p in (HERE, PHASE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
from chaotic_gears import ChaoticBlock
from chaos_lib import permute_indices

# ---------- constants ----------
VOCAB = 512
W = 256       # sequence window
D = 128       # token embedding dim
PRIOR_DIM = 64
BLOCKS = 4


def seeded_block(W, Wl, d, bl, bg, seed):
    """Create a ChaoticBlock with deterministic permutations from seed (П1)."""
    block = ChaoticBlock(W, Wl, d, bl, bg)
    # override permutation buffers with seed-based deterministic maps
    block._sig_l = {t: torch.as_tensor(
        permute_indices(Wl, t + seed * (bl + 3)), dtype=torch.long)
                    for t in range(1, bl + 1)}
    block._sig_g = {t: torch.as_tensor(
        permute_indices(block.Nw, t + seed * (bg + 3)), dtype=torch.long)
                    for t in range(1, bg + 1)}
    return block


class BidirectionalMixer(nn.Module):
    """exp52-style bidirectional mixer, parameterised by d and seed."""
    def __init__(self, d=D, seed=0, blocks=BLOCKS):
        super().__init__()
        Wl = W // 4
        bl, bg = blocks, max(1, blocks // 2)
        self.fwd = seeded_block(W, Wl, d, bl, bg, seed * 2)
        self.bwd = seeded_block(W, Wl, d, bl, bg, seed * 2 + 1)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, x):
        xf = self.fwd(x)
        xb = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.proj(torch.cat([xf, xb], dim=-1))


# ================ DP — Distributed Prior ================
class DPMixer(nn.Module):
    """DP: prior-embedding propagates through SAME permutations as tokens (П1)."""
    def __init__(self, vocab=VOCAB, d=D, prior_dim=PRIOR_DIM, blocks=BLOCKS):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.prior_embed = nn.Embedding(vocab, prior_dim)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        # token mixer (seed=0)
        self.mixer_x = BidirectionalMixer(d=d, seed=0, blocks=blocks)
        # prior mixer with SAME permutation seeds (seed=0) — П1
        self.mixer_p = BidirectionalMixer(d=prior_dim, seed=0, blocks=blocks)
        self.readout = nn.Linear(d + prior_dim, vocab)

    def forward(self, x):
        e = self.embed(x) + self.pos          # [B,W,d]
        p = self.prior_embed(x)               # [B,W,prior_dim]
        x_m = self.mixer_x(e)                 # token mixing
        p_m = self.mixer_p(p)                 # prior propagates through chaos
        return self.readout(torch.cat([x_m, p_m], dim=-1))[:, -1, :]


# ================ DP-noprop (no propagation, static prior) ================
class DPNoPropMixer(nn.Module):
    """DP-noprop: prior embedding exists but does NOT propagate (П2)."""
    def __init__(self, vocab=VOCAB, d=D, prior_dim=PRIOR_DIM, blocks=BLOCKS):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.prior_embed = nn.Embedding(vocab, prior_dim)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.mixer_x = BidirectionalMixer(d=d, seed=0, blocks=blocks)
        # NO mixer_p — prior stays static (per-position)
        self.readout = nn.Linear(d + prior_dim, vocab)

    def forward(self, x):
        e = self.embed(x) + self.pos
        p = self.prior_embed(x)               # no propagation
        x_m = self.mixer_x(e)
        return self.readout(torch.cat([x_m, p], dim=-1))[:, -1, :]


# ================ DP-rand (frozen random prior) ================
class DPRandMixer(nn.Module):
    """DP-rand: prior embedding frozen random (П2)."""
    def __init__(self, vocab=VOCAB, d=D, prior_dim=PRIOR_DIM, blocks=BLOCKS):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        # random frozen prior embedding
        pe = nn.Embedding(vocab, prior_dim)
        nn.init.normal_(pe.weight, std=0.02)
        pe.weight.requires_grad = False
        self.prior_embed = pe
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.mixer_x = BidirectionalMixer(d=d, seed=0, blocks=blocks)
        self.mixer_p = BidirectionalMixer(d=prior_dim, seed=0, blocks=blocks)
        self.readout = nn.Linear(d + prior_dim, vocab)

    def forward(self, x):
        e = self.embed(x) + self.pos
        p = self.prior_embed(x)               # frozen, random, no grad
        x_m = self.mixer_x(e)
        p_m = self.mixer_p(p)
        return self.readout(torch.cat([x_m, p_m], dim=-1))[:, -1, :]


# ================ C-cap (capacity control) ================
class CcapMixer(nn.Module):
    """C-cap: same parameter count as DP, single mixer without prior (П2).
    D is scaled up to match DP's total params."""
    def __init__(self, d_ccap, vocab=VOCAB, blocks=BLOCKS):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_ccap)
        self.pos = nn.Parameter(torch.randn(1, W, d_ccap) * 0.02)
        self.mixer = BidirectionalMixer(d=d_ccap, seed=0, blocks=blocks)
        self.readout = nn.Linear(d_ccap, vocab)

    def forward(self, x):
        e = self.embed(x) + self.pos
        h = self.mixer(e)
        return self.readout(h[:, -1, :])


# ================ PM — Propagating Memory ================
class PropagatingPriorMixer(nn.Module):
    """PM: prior accumulates residually through mixer layers (ТЗ).
    New gates zero-init (П3): at step 0, model ≈ baseline."""
    def __init__(self, vocab=VOCAB, d=D, layers=4, prior_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.layers = nn.ModuleList(
            [BidirectionalMixer(d=d, seed=i, blocks=BLOCKS) for i in range(layers)])
        self.prior_init = nn.Linear(d, prior_dim)
        self.prior_update = nn.ModuleList(
            [nn.Linear(d, prior_dim) for _ in range(layers)])
        # zero init (П3): at step 0, prior_init=0, update=0, model ≈ baseline
        for m in self.prior_update:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.prior_init.weight)
        nn.init.zeros_(self.prior_init.bias)
        self.readout = nn.Linear(d + prior_dim, vocab)

    def forward(self, x):
        h = self.embed(x) + self.pos
        p = torch.tanh(self.prior_init(h))          # [B,W,prior_dim]
        for mixer, update in zip(self.layers, self.prior_update):
            h = mixer(h)
            p = torch.tanh(p + update(h))           # residual prior accumulation
        return self.readout(torch.cat([h, p], dim=-1))[:, -1, :]


# ================ SP — Swarm Prior ================
class PriorAgent(nn.Module):
    """Agent at one scale: local (window) or global (attention-pool over all)."""
    def __init__(self, d, scale, vocab=VOCAB):
        super().__init__()
        self.scale = scale
        if scale == 0:                              # global: attention pool
            self.q = nn.Linear(d, 1)
            self.proj = nn.Linear(d, d // 2)
        else:                                       # local: window conv
            self.conv = nn.Conv1d(d, d // 2, kernel_size=scale, padding=scale//2)
        self.out = nn.Linear(d // 2 if scale else d // 2, d // 2)

    def forward(self, h):
        if self.scale == 0:                         # global
            w = torch.softmax(self.q(h).squeeze(-1), dim=1).unsqueeze(-1)
            g = (h * w).sum(dim=1)
            return self.out(self.proj(g))
        else:                                       # local
            hc = self.conv(h.transpose(1, 2)).transpose(1, 2)
            return self.out(hc.mean(dim=1))


class SwarmPriorMixer(nn.Module):
    """SP: swarm of prior agents at different scales (3, 8, 16, 0=global)."""
    def __init__(self, vocab=VOCAB, d=D, blocks=BLOCKS,
                 scales=(3, 8, 16, 0)):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.mixer = BidirectionalMixer(d=d, seed=0, blocks=blocks)
        self.agents = nn.ModuleList([PriorAgent(d, s) for s in scales])
        agg_dim = d + len(scales) * (d // 2)
        self.readout = nn.Linear(agg_dim, vocab)

    def forward(self, x):
        h = self.mixer(self.embed(x) + self.pos)    # [B,W,d]
        outs = [agent(h) for agent in self.agents]  # list of [B, d//2]
        return self.readout(torch.cat([h[:, -1, :]] + outs, dim=-1))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def build_model(config, vocab=VOCAB, d=D):
    """Build model by config name, return (model, d_ccap_for_cap)."""
    if config == "DP":
        return DPMixer(vocab=vocab, d=d)
    if config == "DP-noprop":
        return DPNoPropMixer(vocab=vocab, d=d)
    if config == "DP-rand":
        return DPRandMixer(vocab=vocab, d=d)
    if config == "C-cap":
        # find d_ccap such that params ≈ DP params
        dp = DPMixer(vocab=vocab, d=d)
        dp_params = count_params(dp)
        lo, hi = 64, 512
        while lo < hi:
            mid = (lo + hi) // 2
            m = CcapMixer(mid, vocab=vocab)
            if count_params(m) < dp_params:
                lo = mid + 1
            else:
                hi = mid
        d_ccap = max(64, lo)
        return CcapMixer(d_ccap, vocab=vocab)
    if config == "PM":
        return PropagatingPriorMixer(vocab=vocab, d=d)
    if config == "SP":
        return SwarmPriorMixer(vocab=vocab, d=d)
    raise ValueError(f"Unknown config: {config}")


if __name__ == "__main__":
    # quick sanity: param counts
    for cfg in ["DP", "DP-noprop", "DP-rand", "C-cap", "PM", "SP"]:
        m = build_model(cfg, vocab=VOCAB)
        print(f"  {cfg:15s}: {count_params(m):>10,} params")