"""
phase3.py — Phase 3: scaling (exp_11), attention comparison (exp_12),
ablation (exp_13).

exp_11 — accuracy vs N, FLOPs vs N for ChaoticMixer vs FullAttn.
exp_12 — chaotic routing vs attention at matched N.
exp_13 — ablation of ChaoticMixer components on task A.

Run:  python phase3.py <exp>   (exp: scaling | ablation)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaotic_torch import ChaoticMixer, ChaoticAttnReadout, FullAttn
from chaos_lib import flops_attention, flops_chaotic_step, flops_gsf_pooled
from toy_data import taskA_batch

torch.manual_seed(0)
np.random.seed(0)

BASE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.join(BASE, "experiments")


def scaling_task_batch(bs, N, V=32, rng=None):
    """N tokens, each with a UNIQUE key (0..N-1) and a value (0..V-1).
    Query at position 0 asks for a random key's value.
    d_in = N + V (key one-hot + value one-hot)."""
    rng = rng if rng is not None else np.random.default_rng()
    d_in = N + V
    inp = np.zeros((bs, N, d_in), dtype=np.float32)
    vals = rng.integers(0, V, size=(bs, N))
    for i in range(N):
        inp[:, i, i] = 1.0  # unique key one-hot
    inp[:, :, N:] = np.eye(V)[vals]
    qkey = rng.integers(0, N, size=(bs,))
    qpos = N - 1  # query at the END (avoids the cat map's fixed point at (0,0))
    inp[np.arange(bs), qpos, :N] = 0.0
    inp[np.arange(bs), qpos, qkey] = 1.0   # query key at last position
    inp[np.arange(bs), qpos, N:] = 0.0     # no value at query
    qmask = np.zeros((bs, N), dtype=np.float32)
    qmask[:, qpos] = 1.0
    target = vals[np.arange(bs), qkey].astype(np.int64)
    return (torch.tensor(inp), torch.tensor(qmask), torch.tensor(target))


def train_quick(model, task, bs=64, epochs=400, lr=1e-3, extra=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        x, q, y = task(bs, **extra)
        opt.zero_grad()
        out = model(x, q)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
    return model


def eval_acc(model, task, extra, n_batches=40, bs=128):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, q, y = task(bs, **extra)
            out = model(x, q)
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


def run_scaling():
    results = {"N": [], "chaotic": {}, "chaotic_attnread": {}, "fullattn": {}}
    for N in [9, 16, 25, 64, 121, 256]:
        epochs = 300 if N > 121 else 400
        d_in = N + 32
        d_out = 32
        n_blocks = max(3, int(np.ceil(np.log2(N))) + 2)
        t0 = time.perf_counter()
        cm = ChaoticMixer(N, d_in, d_out, d_model=32, n_blocks=n_blocks)
        train_quick(cm, scaling_task_batch, extra={"N": N}, epochs=epochs)
        acc_cm = eval_acc(cm, scaling_task_batch, {"N": N})
        t_cm = time.perf_counter() - t0

        t0 = time.perf_counter()
        cr = ChaoticAttnReadout(N, d_in, d_out, d_model=32, n_blocks=n_blocks)
        train_quick(cr, scaling_task_batch, extra={"N": N}, epochs=epochs)
        acc_cr = eval_acc(cr, scaling_task_batch, {"N": N})
        t_cr = time.perf_counter() - t0

        t0 = time.perf_counter()
        fa = FullAttn(N, d_in, d_out, d_model=32, n_layers=2)
        train_quick(fa, scaling_task_batch, extra={"N": N}, epochs=epochs)
        acc_fa = eval_acc(fa, scaling_task_batch, {"N": N})
        t_fa = time.perf_counter() - t0

        flops_cm = n_blocks * flops_chaotic_step(N, 32)
        flops_cr = n_blocks * flops_chaotic_step(N, 32) + 2 * N * 32 * 32
        flops_fa = 2 * flops_attention(N, 32) + 2 * flops_gsf_pooled(N, 32)

        results["N"].append(N)
        results["chaotic"][str(N)] = {"acc": acc_cm, "flops": flops_cm,
                                      "train_s": t_cm, "blocks": n_blocks}
        results["chaotic_attnread"][str(N)] = {"acc": acc_cr, "flops": flops_cr,
                                               "train_s": t_cr, "blocks": n_blocks}
        results["fullattn"][str(N)] = {"acc": acc_fa, "flops": flops_fa,
                                       "train_s": t_fa}
        print(f"[11] N={N:4d} chaotic={acc_cm:.3f} chaotic+attnread={acc_cr:.3f} "
              f"fullattn={acc_fa:.3f}  flops_c={flops_cm:.2e} flops_a={flops_cr:.2e} "
              f"flops_f={flops_fa:.2e}")
    outdir = os.path.join(HERE, "exp_11_scaling")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


# ============ exp_13: ablation ============

class ChaoticPermOnly(nn.Module):
    """Ablation -coupling: permutation only, no value mixing.
    Cannot learn any cross-token task (support is conserved)."""
    def __init__(self, n_tokens, d_in, d_out, d_model=32, n_blocks=6):
        super().__init__()
        self.mixer = ChaoticMixer(n_tokens, d_in, d_out, d_model=d_model,
                                  n_blocks=n_blocks)

    def forward(self, x, query_mask=None):
        B, N, _ = x.shape
        h = self.mixer.embed(x) + self.mixer.pos_embed
        for step, block in enumerate(self.mixer.blocks):
            # permutation only — bypass the coupling via gate -> 0
            with torch.no_grad():
                pass
            # simulate: set gate to 0 by replacing block output
            sigma = self.mixer._sigmas[step + 1]
            h = h[:, sigma, :]
        m = query_mask.float().unsqueeze(-1)
        out = (h * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        return self.mixer.readout(out)


class ChaoticNoPerm(nn.Module):
    """Ablation -permutation: coupling only (fixed even-odd pairs)."""
    def __init__(self, n_tokens, d_in, d_out, d_model=32, n_blocks=6):
        super().__init__()
        self.mixer = ChaoticMixer(n_tokens, d_in, d_out, d_model=d_model,
                                  n_blocks=n_blocks)

    def forward(self, x, query_mask=None):
        B, N, _ = x.shape
        h = self.mixer.embed(x) + self.mixer.pos_embed
        for step, block in enumerate(self.mixer.blocks):
            # coupling only (no permutation): use identity sigma
            g = torch.sigmoid(block.gate)
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, -1)
        m = query_mask.float().unsqueeze(-1)
        out = (h * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        return self.mixer.readout(out)


def run_ablation():
    N, d_in, d_out = 16, d_in_for("A"), 16
    results = {}
    variants = {
        "full_gsf": ChaoticMixer(N, d_in, d_out, d_model=32, n_blocks=6,
                                 gsf_hidden=64),
        "no_gsf": ChaoticMixer(N, d_in, d_out, d_model=32, n_blocks=6),
        "shallow_3": ChaoticMixer(N, d_in, d_out, d_model=32, n_blocks=3),
        "deep_10": ChaoticMixer(N, d_in, d_out, d_model=32, n_blocks=10),
        "perm_only": ChaoticPermOnly(N, d_in, d_out, d_model=32, n_blocks=6),
        "no_perm": ChaoticNoPerm(N, d_in, d_out, d_model=32, n_blocks=6),
    }
    for name, model in variants.items():
        t0 = time.perf_counter()
        train_quick(model, taskA_batch, extra={"N": N}, epochs=400)
        acc = eval_acc(model, taskA_batch, {"N": N})
        t = time.perf_counter() - t0
        results[name] = {"acc": acc, "params": sum(p.numel() for p in model.parameters()),
                         "train_s": t}
        print(f"[13] {name:12s} acc={acc:.3f} params={results[name]['params']}")
    outdir = os.path.join(HERE, "exp_13_ablation")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    from toy_data import d_in_for
    exp = sys.argv[1] if len(sys.argv) > 1 else "all"
    if exp in ("scaling", "all"):
        print("=== exp_11: scaling ===")
        run_scaling()
    if exp in ("ablation", "all"):
        print("=== exp_13: ablation ===")
        run_ablation()
