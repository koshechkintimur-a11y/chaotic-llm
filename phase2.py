"""
phase2.py — Phase 2 experiments: GSF steering, adaptive depth,
error controller, reversibility under control.

exp_06 (GSF):  (a) combinatorial: can a small control signal steer the
               cat map to a target in a large space?  Random vs Greedy(cheat)
               vs content-based control.
               (b) learned: Chaotic+GSF vs Chaotic on task A (from
               train_models results — see results.json).
exp_07 (adaptive depth): accuracy vs number of blocks T at inference.
exp_08 (error controller): deep-supervision readout, early stable exit.
exp_09 (reversibility under control): decode(encode(X)) error with fixed
               gate vs GSF gate vs adaptive depth.

Run:  python phase2.py [outdir]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import arnold_map, period, permute_indices
from chaotic_torch import ChaoticMixer
from toy_data import taskA_batch, d_in_for

torch.manual_seed(0)
np.random.seed(0)

BASE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.join(BASE, "experiments")


# ============================================================
# exp_06a: combinatorial steering of the affine cat map
# ============================================================

def affine_steer(N_grid, T, target, ctrl):
    """Affine dynamics: x_{t+1} = A x_t + c_t (mod N_grid).
    ctrl: 'random' | 'greedy' (knows target) | 'zero'.
    Returns (success, steps_to_target).
    Target is a single grid cell; success = the trajectory visits it.
    """
    rng = np.random.default_rng(0)
    pos = np.array([0, 0], dtype=np.int64)  # start
    visited = set()
    for t in range(1, T + 1):
        if ctrl == 'random':
            c = rng.integers(0, N_grid, size=2)
        elif ctrl == 'greedy':
            # one-step correction toward target (cheating: knows target)
            nxt = arnold_map(pos.reshape(1, 2), N_grid, 1).reshape(2)
            c = (target - nxt) % N_grid
        elif ctrl == 'zero':
            c = np.array([0, 0])
        pos = (arnold_map(pos.reshape(1, 2), N_grid, 1).reshape(2) + c) % N_grid
        visited.add((int(pos[0]), int(pos[1])))
        if (int(pos[0]), int(pos[1])) == (int(target[0]), int(target[1])):
            return True, t
    return False, None


def run_exp06a():
    results = {"grids": {}}
    for Ng in [32, 64, 128, 256, 512, 1024]:
        T = min(200, int(Ng))
        n_trials = 50
        rng = np.random.default_rng(1)
        row = {}
        for ctrl in ['random', 'greedy', 'zero']:
            succ = 0
            steps = []
            for _ in range(n_trials):
                target = rng.integers(0, Ng, size=2)
                ok, t = affine_steer(Ng, T, target, ctrl)
                succ += ok
                if ok:
                    steps.append(t)
            row[ctrl] = {
                "success_rate": succ / n_trials,
                "mean_steps_if_success": float(np.mean(steps)) if steps else None,
            }
        results["grids"][str(Ng)] = row
        print(f"[06a] N_grid={Ng}: " +
              "  ".join(f"{k}={v['success_rate']:.2f}" for k, v in row.items()))
    return results


# ============================================================
# exp_07: accuracy vs depth (adaptive depth ablation)
# ============================================================

def run_exp07():
    """Train ChaoticMixer on task A; evaluate accuracy with the first T
    blocks only (T = 1..n_blocks).  This gives the accuracy-compute curve."""
    N, d_in, d_out = 16, d_in_for("A"), 16
    model = ChaoticMixer(N, d_in, d_out, d_model=32, n_blocks=8)
    # train
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(400):
        x, q, y = taskA_batch(64, N=N)
        opt.zero_grad()
        out = model(x, q)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
    # evaluate at each truncation depth
    acc_by_T = {}
    model.eval()
    with torch.no_grad():
        for T in range(1, 9):
            correct = total = 0
            for _ in range(40):
                x, q, y = taskA_batch(128, N=N)
                # truncate forward: manually run T blocks
                h = model.embed(x) + model.pos_embed
                for step in range(T):
                    h = model.blocks[step](h, step + 1)
                m = q.float().unsqueeze(-1)
                out = (h * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
                logits = model.readout(out)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.numel()
            acc_by_T[str(T)] = correct / total
    model.train()
    print("[07] accuracy by depth:", {k: round(v, 3) for k, v in acc_by_T.items()})
    return {"acc_by_depth": acc_by_T}


# ============================================================
# exp_08: error controller (deep supervision + early stable exit)
# ============================================================

class ChaoticDeepSup(nn.Module):
    """ChaoticMixer with a readout at EVERY block (deep supervision).

    At inference, pick the earliest depth whose prediction is stable
    (argmax doesn't change for 2 consecutive blocks) — the 'error
    controller' exits early when E_{t+1} >= E_t (no improvement)."""
    def __init__(self, n_tokens, d_in, d_out, d_model=32, n_blocks=8):
        super().__init__()
        self.mixer = ChaoticMixer(n_tokens, d_in, d_out, d_model=d_model,
                                  n_blocks=n_blocks)
        self.readouts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, d_out))
            for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

    def forward(self, x, query_mask=None, return_all=False):
        B, N, _ = x.shape
        h = self.mixer.embed(x) + self.mixer.pos_embed
        all_logits = []
        for step, block in enumerate(self.mixer.blocks):
            h = block(h, step + 1)
            m = query_mask.float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
            all_logits.append(self.readouts[step](pooled))
        if return_all:
            return all_logits
        return all_logits[-1]

    def forward_early(self, x, query_mask=None, stability=2):
        """Inference with early exit: stop when argmax stable for `stability`
        consecutive blocks, or at the end.  Returns (logits, exit_depth)."""
        B, N, _ = x.shape
        h = self.mixer.embed(x) + self.mixer.pos_embed
        prev_pred = None
        stable = 0
        exit_depth = self.n_blocks
        last_logits = None
        for step, block in enumerate(self.mixer.blocks):
            h = block(h, step + 1)
            m = query_mask.float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
            logits = self.readouts[step](pooled)
            last_logits = logits
            pred = logits.argmax(-1)
            if prev_pred is not None and (pred == prev_pred).float().mean().item() > 0.99:
                stable += 1
            else:
                stable = 0
            prev_pred = pred
            if stable >= stability:
                exit_depth = step + 1
                break
        return last_logits, exit_depth


def run_exp08():
    N, d_in, d_out = 16, d_in_for("A"), 16
    model = ChaoticDeepSup(N, d_in, d_out, d_model=32, n_blocks=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(400):
        x, q, y = taskA_batch(64, N=N)
        opt.zero_grad()
        all_logits = model(x, q, return_all=True)
        loss = sum(lossf(lo, y) for lo in all_logits) / len(all_logits)
        loss.backward()
        opt.step()
    # evaluate: fixed depth 8 vs early exit
    model.eval()
    acc_fixed = 0
    acc_early = 0
    total = 0
    depths = []
    with torch.no_grad():
        for _ in range(50):
            x, q, y = taskA_batch(128, N=N)
            out = model(x, q)
            acc_fixed += (out.argmax(-1) == y).sum().item()
            lo, dep = model.forward_early(x, q)
            acc_early += (lo.argmax(-1) == y).sum().item()
            depths.append(dep)
            total += y.numel()
    model.train()
    results = {
        "acc_fixed_depth8": acc_fixed / total,
        "acc_early_exit": acc_early / total,
        "mean_exit_depth": float(np.mean(depths)),
        "compute_saved_frac": 1.0 - float(np.mean(depths)) / 8,
    }
    print("[08]", {k: round(v, 3) if isinstance(v, float) else v
                   for k, v in results.items()})
    return results


# ============================================================
# exp_09: reversibility under control
# ============================================================

def inverse_block(y, step, g, gsf_control=None):
    """Invert one chaotic block: first invert coupling, then permutation.
    y: (B, N, d) state after the block.  g: coupling gate scalar.
    gsf_control: gates used in the forward pass (must be stored)."""
    B, N, d = y.shape
    # invert symmetric coupling
    even = y[:, 0::2, :]
    odd = y[:, 1::2, :]
    if gsf_control is not None:
        g_even = gsf_control[:, 0::2, :]
        g_odd = gsf_control[:, 1::2, :]
    else:
        g_even = g_odd = g
    g2 = g_even * g_odd
    odd_orig = (odd - g_odd * even) / (1 - g2)
    even_orig = even - g_even * odd_orig
    x = torch.stack([even_orig, odd_orig], dim=2).reshape(B, N, d)
    # invert permutation
    sigma = permute_indices(N, step)
    inv_sigma = np.argsort(sigma)
    inv_t = torch.as_tensor(inv_sigma, dtype=torch.long, device=y.device)
    return x[:, inv_t, :]


def run_exp09():
    N, d_in, d_out = 16, d_in_for("A"), 16
    results = {}

    # (a) fixed gate, no GSF
    model = ChaoticMixer(N, d_in, d_out, d_model=8, n_blocks=4, gsf_hidden=None)
    model.eval()
    with torch.no_grad():
        x, q, y = taskA_batch(8, N=N)
        h = model.embed(x) + model.pos_embed
        states = [h]
        for step, block in enumerate(model.blocks):
            h = block(h, step + 1)
            states.append(h.clone())
        # decode from the end
        hd = states[-1].clone()
        for step in range(4, 0, -1):
            block = model.blocks[step - 1]
            g = torch.sigmoid(block.gate)
            hd = inverse_block(hd, step, g)
        err = (hd - states[0]).abs().max().item()
    results["no_gsf_decode_max_err"] = err
    print(f"[09] no-GSF decode error: {err:.2e}")

    # (b) with GSF: controls depend on states -> decode needs stored gates
    model_g = ChaoticMixer(N, d_in, d_out, d_model=8, n_blocks=4, gsf_hidden=16)
    model_g.eval()
    with torch.no_grad():
        x, q, y = taskA_batch(8, N=N)
        h = model_g.embed(x) + model_g.pos_embed
        states = [h]
        gates = []
        for step, block in enumerate(model_g.blocks):
            pooled = h.mean(dim=1, keepdim=True)
            ctrl = model_g.gsf(h + pooled)
            gates.append(ctrl.clone())
            h = block(h, step + 1, gsf_control=ctrl)
            states.append(h.clone())
        # decode WITH stored gates
        hd = states[-1].clone()
        for step in range(4, 0, -1):
            block = model_g.blocks[step - 1]
            g = torch.sigmoid(block.gate)
            hd = inverse_block(hd, step, g, gsf_control=gates[step - 1])
        err_with = (hd - states[0]).abs().max().item()
        # decode WITHOUT stored gates (recompute from wrong state -> wrong gates)
        hd2 = states[-1].clone()
        for step in range(4, 0, -1):
            block = model_g.blocks[step - 1]
            g = torch.sigmoid(block.gate)
            # recompute gates from the CURRENT (corrupted) state
            pooled = hd2.mean(dim=1, keepdim=True)
            ctrl = model_g.gsf(hd2 + pooled)
            hd2 = inverse_block(hd2, step, g, gsf_control=ctrl)
        err_without = (hd2 - states[0]).abs().max().item()
    results["gsf_decode_err_with_stored_gates"] = err_with
    results["gsf_decode_err_recomputed_gates"] = err_without
    print(f"[09] GSF decode error (stored gates): {err_with:.2e}")
    print(f"[09] GSF decode error (recomputed gates): {err_without:.2e}")

    # (c) adaptive depth: decode requires knowing T
    results["adaptive_depth_note"] = (
        "adaptive depth requires storing the exit depth T; "
        "decode is defined only for the exact T used in encode")
    return results


if __name__ == "__main__":
    outdir = os.path.join(HERE, sys.argv[1]) if len(sys.argv) > 1 else HERE
    outdir = os.path.join(HERE, "exp_06_gsf") if len(sys.argv) <= 1 else \
        os.path.join(HERE, sys.argv[1])
    os.makedirs(outdir, exist_ok=True)

    all_results = {}
    print("=== exp_06a: affine steering ===")
    all_results["exp06a_steering"] = run_exp06a()
    print("\n=== exp_07: adaptive depth ===")
    all_results["exp07_adaptive_depth"] = run_exp07()
    print("\n=== exp_08: error controller ===")
    all_results["exp08_error_controller"] = run_exp08()
    print("\n=== exp_09: reversibility under control ===")
    all_results["exp09_reversibility"] = run_exp09()

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved to {outdir}/results.json")