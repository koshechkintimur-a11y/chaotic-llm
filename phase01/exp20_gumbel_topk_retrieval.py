"""exp20_gumbel_topk_retrieval.py — Step 8: learnable selection on retrieval.

The decisive test of "chaotic dynamics generates a cheap structured proposal
space for selective computation": on Task A (associative recall, where content
selection is CRITICAL), can a LEARNABLE temperature-softmax selector find the
correct key tokens from the chaotic state?

vs exp19: temperature annealing (softmax over ALL tokens, τ 2.0→0.2) gives the
selector a REAL gradient for the choice itself (exp19's hard top-K mask had no
selection gradient).

Metrics:
  acc            — retrieval accuracy
  recall@K       — is at least one correct key token in the top-K?
  (K=2: the answer needs 1 of the 2 positions carrying the query key)

Baselines:
  FullAttn (transformer) — upper bound
  ChaoticAttnReadout     — chaotic + attention over all N (Phase 3: 0.92)
  ChaoticLocal           — chaotic + local readout (no attention)
  ChaoticTopK (this)     — chaotic + learned selector + attention over K
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices
from toy_data import taskA_batch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp20_gumbel_topk_retrieval")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

N = 16          # tokens (4x4 grid)
K_KEYS = 8
V_VAL = 16
D_IN = K_KEYS + V_VAL
D = 64
BLOCKS = 6      # log2(16)+2
BATCH = 128
STEPS = 3000
LR = 1e-3
K_SEL = 2       # top-K selection (2 positions carry the query key)
TAU_START, TAU_END = 2.0, 0.2


class ChaoticMixer(nn.Module):
    def __init__(self, N, d, blocks):
        super().__init__()
        self.N, self.d = N, d
        self.gates = nn.Parameter(torch.zeros(blocks))
        self._sig = {t: torch.as_tensor(permute_indices(N, t), dtype=torch.long)
                     for t in range(1, blocks + 1)}

    def forward(self, h):
        B, N, d = h.shape
        for t in range(1, len(self.gates) + 1):
            h = h[:, self._sig[t].to(h.device), :]
            g = torch.sigmoid(self.gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h


class FullAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_IN, D)
        self.qkv = nn.Linear(D, D * 3)
        self.out = nn.Linear(D, V_VAL)

    def forward(self, inp):
        B, N, _ = inp.shape
        h = self.proj(inp)
        qkv = self.qkv(h).reshape(B, N, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        w = F.softmax((q @ k.transpose(-2, -1)) / (D ** 0.5), -1)
        h = w @ v
        return self.out(h[:, -1, :])


class ChaoticAttnReadout(nn.Module):
    """chaotic mix + attention readout over all N (Phase 3 winner)."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_IN, D)
        self.mix = ChaoticMixer(N, D, BLOCKS)
        self.q = nn.Linear(D, D)
        self.kv = nn.Linear(D, D * 2)
        self.out = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, V_VAL))

    def forward(self, inp):
        B, N, _ = inp.shape
        h = self.mix(self.proj(inp))
        q = self.q(h[:, -1:, :])
        kv = self.kv(h).reshape(B, N, 2, D).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        w = F.softmax((q @ k.transpose(-2, -1)) / (D ** 0.5), -1)
        att = (w @ v).squeeze(1)
        return self.out(torch.cat([h[:, -1, :], att], dim=-1))


class ChaoticLocal(nn.Module):
    """chaotic mix + local readout (no attention anywhere)."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_IN, D)
        self.mix = ChaoticMixer(N, D, BLOCKS)
        self.out = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, V_VAL))

    def forward(self, inp):
        B, N, _ = inp.shape
        h = self.mix(self.proj(inp))
        g = h.mean(dim=1)
        return self.out(torch.cat([h[:, -1, :], g], dim=-1))


class ChaoticTopK(nn.Module):
    """chaotic mix + LEARNABLE selector (temperature softmax) + attn over K.

    Training: softmax(scores/τ) over ALL tokens, τ annealed down → the selector
    gets a real gradient for the choice.  Eval: hard top-K mask.
    """
    def __init__(self, k_sel=K_SEL):
        super().__init__()
        self.k_sel = k_sel
        self.proj = nn.Linear(D_IN, D)
        self.mix = ChaoticMixer(N, D, BLOCKS)
        self.selector = nn.Linear(D, 1, bias=False)
        self.q = nn.Linear(D, D)
        self.kv = nn.Linear(D, D * 2)
        self.out = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, V_VAL))

    def forward(self, inp, tau=1.0, hard=False):
        B, N, _ = inp.shape
        h = self.mix(self.proj(inp))
        scores = self.selector(h).squeeze(-1)         # (B, N)
        if hard:
            # hard top-K mask (eval)
            _, idx = scores.topk(self.k_sel, dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, idx, 1.0)
            w = mask / mask.sum(-1, keepdim=True)
        else:
            w = F.softmax(scores / tau, dim=-1)       # temperature softmax
        q = self.q(h[:, -1:, :])
        kv = self.kv(h).reshape(B, N, 2, D).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        w = w.unsqueeze(1)
        att = (w @ v).squeeze(1)                      # attention over (soft) top-K
        return self.out(torch.cat([h[:, -1, :], att], dim=-1))


def train(model, steps=STEPS, tau_sched=True, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for s in range(steps):
        inp, qmask, target = taskA_batch(BATCH, N=N, K=K_KEYS, V=V_VAL)
        tau = max(TAU_END, TAU_START * (1 - s / steps)) if tau_sched else 0.5
        opt.zero_grad()
        logits = model(inp, tau=tau) if hasattr(model, "selector") else model(inp)
        loss = lossf(logits, target)
        loss.backward()
        opt.step()
        if s % 1000 == 0 and s > 0:
            print(f"  [{tag} {s:,}] loss={loss.item():.3f} τ={tau:.2f} ({time.time()-t0:.0f}s)")
    return loss.item()


@torch.no_grad()
def evaluate(model, n_batches=100, hard=False):
    model.eval()
    acc = 0
    n = 0
    for _ in range(n_batches):
        inp, qmask, target = taskA_batch(64, N=N, K=K_KEYS, V=V_VAL)
        if hasattr(model, "selector"):
            logits = model(inp, tau=TAU_END, hard=hard)
        else:
            logits = model(inp)
        acc += (logits.argmax(-1) == target).sum().item()
        n += len(target)
    model.train()
    return acc / n


@torch.no_grad()
def selection_quality(model, n_batches=50, k=K_SEL):
    """recall@k: is at least one position carrying the query key in top-k?"""
    model.eval()
    hit = 0
    n = 0
    for _ in range(n_batches):
        inp, qmask, target = taskA_batch(64, N=N, K=K_KEYS, V=V_VAL)
        h = model.mix(model.proj(inp))
        scores = model.selector(h).squeeze(-1)
        _, idx = scores.topk(k, dim=-1)
        # positions carrying the query key: key one-hot at columns 0..K_KEYS
        keys = inp[:, :, :K_KEYS].argmax(-1)          # (B, N)
        qkey = keys[:, -1]                             # query key = key at last pos
        for b in range(len(qkey)):
            correct_pos = (keys[b, :-1] == qkey[b]).nonzero().flatten()
            if (idx[b][:, None] == correct_pos[None, :]).any():
                hit += 1
            n += 1
    model.train()
    return hit / n


results = {"N": N, "K_keys": K_KEYS, "V_val": V_VAL, "D": D, "blocks": BLOCKS,
           "steps": STEPS, "batch": BATCH, "k_sel": K_SEL,
           "tau": [TAU_START, TAU_END]}

models = [
    ("FullAttn", FullAttn(), False),
    ("ChaoticAttnReadout", ChaoticAttnReadout(), False),
    ("ChaoticLocal", ChaoticLocal(), False),
    ("ChaoticTopK", ChaoticTopK(), True),
]

for name, model, is_tk in models:
    print(f"\n=== {name} ===")
    n_params = sum(p.numel() for p in model.parameters())
    ckpt = os.path.join(OUT, f"{name}.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded ({n_params:,} params)")
    else:
        print(f"params: {n_params:,}")
        train(model, tag=name, tau_sched=is_tk)
        torch.save(model.state_dict(), ckpt)
    r = {"acc": evaluate(model)}
    if is_tk:
        r["acc_hard_topk"] = evaluate(model, hard=True)
        r["recall@K"] = selection_quality(model)
    print(f"{name}: {r}")
    results[name] = {**r, "params": n_params}

import json
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
