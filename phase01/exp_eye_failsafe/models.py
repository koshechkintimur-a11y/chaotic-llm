"""models.py — model classes + config for the Eye failsafe experiment.

Extracted from experiment.py so analysis can import without triggering training.
Keep this in sync with experiment.py.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
sys.path.insert(0, HERE); sys.path.insert(0, PHASE); sys.path.insert(0, REPO)

from chaotic_gears import ChaoticBlock

# ---------------- config (MUST match experiment.py) ----------------
VOCAB = 512
W = 256
D = 128
R = 4          # E1 routes
K = 4          # E2 groups
BLOCKS = 4     # per-direction blocks in each bidirectional mixer
EPS = 0.25     # eps-floor for learned Eye


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


def make_bpe(text, vs=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vs, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i+100000] for i in range(0, len(text), 100000)], trainer=tr)
    return tok


def build_tokenizer():
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), 2_000_000)
    return make_bpe(train_text)


def seeded_block(W, Wl, d, bl, bg, seed):
    from chaos_lib import permute_indices
    block = ChaoticBlock(W, Wl, d, bl, bg)
    block._sig_l = {t: torch.as_tensor(permute_indices(Wl, t + seed * (bl + 3)),
                                       dtype=torch.long) for t in range(1, bl + 1)}
    block._sig_g = {t: torch.as_tensor(permute_indices(block.Nw, t + seed * (bg + 3)),
                                       dtype=torch.long) for t in range(1, bg + 1)}
    return block


class BidirectionalMixer(nn.Module):
    def __init__(self, seed=0, blocks=BLOCKS):
        super().__init__()
        Wl = W // 4
        bl, bg = blocks, max(1, blocks // 2)
        self.fwd = seeded_block(W, Wl, D, bl, bg, seed * 2)
        self.bwd = seeded_block(W, Wl, D, bl, bg, seed * 2 + 1)
        self.proj = nn.Linear(2 * D, D)

    def forward(self, x):
        xf = self.fwd(x)
        xb = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.proj(torch.cat([xf, xb], dim=-1))


class LMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        self.readout = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, VOCAB))

    def head(self, h):
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))

    def embed_pos(self, x):
        return self.embed(x) + self.pos


class EyeModulatorLM(nn.Module):
    def __init__(self, mode="l"):
        super().__init__()
        self.head = LMHead()
        self.routes = nn.ModuleList([BidirectionalMixer(seed=i) for i in range(R)])
        self.mode = mode
        self.eye = nn.Sequential(nn.Linear(D, D // 8), nn.ReLU(), nn.Linear(D // 8, R))
        if mode == "u":
            self.eps = 1.0
        else:
            self.eps = EPS
            if mode == "l":
                nn.init.zeros_(self.eye[-1].weight)
                nn.init.zeros_(self.eye[-1].bias)
            else:
                nn.init.normal_(self.eye[-1].weight, 0, 0.3)
                nn.init.normal_(self.eye[-1].bias, 0, 0.3)
                self.eye.requires_grad_(False)
        self._w = None

    def forward(self, x):
        e = self.head.embed_pos(x)
        outs = torch.stack([r(e) for r in self.routes], dim=-2)
        if self.mode == "u":
            w = torch.full((outs.shape[0], outs.shape[1], R), 1.0 / R, device=outs.device)
        else:
            logits = self.eye(e)
            w = (1 - self.eps) * torch.softmax(logits, dim=-1) + self.eps / R
        self._w = w
        h = (outs * w.unsqueeze(-1)).sum(-2)
        return self.head.head(h)


def top1_with_capacity(logits, C):
    B, W_, K_ = logits.shape
    assign = torch.full((B, W_), -1, dtype=torch.long, device=logits.device)
    for k in range(K_):
        best = logits.argmax(-1)
        mask = (best == k)
        scores = logits[:, :, k].masked_fill(~mask, -1e9)
        topc = scores.topk(min(C, W_), dim=-1).indices
        assign.scatter_(1, topc, k)
    unassigned = (assign == -1)
    if unassigned.any():
        assign = torch.where(unassigned, logits.argmax(-1), assign)
    return assign


class EyeGroupLM(nn.Module):
    def __init__(self, mode="l"):
        super().__init__()
        self.head = LMHead()
        self.base = BidirectionalMixer(seed=0)
        self.groups = nn.ModuleList([BidirectionalMixer(seed=i + 1) for i in range(K)])
        self.mode = mode
        self.eye = nn.Linear(D, K)
        self.gate = nn.Parameter(torch.zeros(1))
        self.C = W // K
        if mode == "r":
            nn.init.normal_(self.eye.weight, 0, 0.3)
            self.eye.requires_grad_(False)
        else:
            nn.init.zeros_(self.eye.weight)
            nn.init.zeros_(self.eye.bias)
        self._assign = None
        self._logits = None

    def _group_mix(self, e, assign):
        B, W_, D = e.shape
        out = torch.zeros_like(e)
        for k in range(self.K if hasattr(self, 'K') else K):
            idx = (assign == k)
            for b in range(B):
                pos = idx[b].nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                chunk = e[b, pos].unsqueeze(0)
                mixed = self.groups[k](chunk)
                out[b, pos] = mixed[0, :pos.numel()]
        return out

    def forward(self, x):
        e = self.head.embed_pos(x)
        base = self.base(e)
        logits = self.eye(e)
        self._logits = logits
        assign = top1_with_capacity(logits, self.C)
        self._assign = assign
        extra = self._group_mix(e, assign)
        h = base + torch.sigmoid(self.gate) * extra
        return self.head.head(h)

    def balance_loss(self):
        if self._logits is None:
            return torch.tensor(0.0)
        P = torch.softmax(self._logits, dim=-1)
        f = P.mean(dim=1)
        return K * (f * P.mean(dim=0)).sum()
