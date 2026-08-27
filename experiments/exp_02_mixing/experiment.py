"""exp_02_mixing/experiment.py

Experiment 0.3 — Chaos vs mixing.

QUESTIONS
  - Does the cat-map permutation "spread" a perturbation between two
    near-identical states (X vs X+eps)?
  - How does the state distance d_t behave under different dynamics?
  - Does a pure permutation mix VALUES at all?

KEY FINDINGS (measured)
  1. Any permutation (Arnold, fixed, random, translation) conserves the L2
     distance between two state vectors: d_t is constant.  Permutations are
     isometries; they do NOT mix values.
  2. Value mixing requires a combine operation (coupling).  But the naive
     additive coupling X_dst += 0.5*X_src is numerically UNSTABLE: values
     grow like (1.5)^t (measured: 1e17 after 200 steps).
  3. The cat map fixes the origin cell (0,0): a token at the origin never
     moves.  This is a genuine structural defect (the linear map fixes 0).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from chaos_lib import (arnold_map, coupling_mix, square_grid_size,
                       permute_sequence)

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)

N_TOKENS = 256
N_GRID = square_grid_size(N_TOKENS)
T_MAX = 60
EPS = 0.01

results = {"N_tokens": N_TOKENS, "N_grid": N_GRID, "T_max": T_MAX}

# ===== Part 1: state distance d_t, SAME map applied to both states =====
X = np.zeros((N_TOKENS, 1))
X[0, 0] = 1.0
X_pert = X.copy()
X_pert[5, 0] += EPS

def apply_perm(X, sigma):
    return X[sigma]

# schedule generators: return the permutation for step t (1-indexed)
fixed_perm = RNG.permutation(N_TOKENS)

def run_dist(schedule):
    Xc, Xpc = X.copy(), X_pert.copy()
    dists = []
    for t in range(1, T_MAX + 1):
        if schedule == "arnold":
            Xc = permute_sequence(Xc, 1)
            Xpc = permute_sequence(Xpc, 1)
        elif schedule == "fixed":
            Xc = Xc[fixed_perm]
            Xpc = Xpc[fixed_perm]
        elif schedule == "random":
            # SAME random perm applied to both states each step
            p = RNG.permutation(N_TOKENS)
            Xc = Xc[p]
            Xpc = Xpc[p]
        elif schedule == "translation":
            Xc = np.roll(Xc, 1, axis=0)
            Xpc = np.roll(Xpc, 1, axis=0)
        elif schedule == "arnold_coupling":
            Xc = permute_sequence(Xc, 1)
            Xc = coupling_mix(Xc, t)
            Xpc = permute_sequence(Xpc, 1)
            Xpc = coupling_mix(Xpc, t)
        dists.append(float(np.linalg.norm(Xc - Xpc)))
    return dists

state_dist = {}
for sched in ["arnold", "fixed", "random", "translation", "arnold_coupling"]:
    state_dist[sched] = run_dist(sched)
results["state_distance"] = state_dist

# ===== Part 2: position decorrelation of a NON-origin token =====
# Track a token NOT at the origin (token 1 -> grid (0,1)).
pos = np.array([0, 1], dtype=np.int64)
positions = [pos.copy()]
for t in range(1, 40):
    pos = arnold_map(pos.reshape(1, 2), N_GRID, 1).reshape(2)
    positions.append(pos.copy())
positions = np.array(positions)
results["origin_token_tracking"] = {
    "token0_positions": [0, 0],          # (0,0) is a FIXED POINT of A
    "token1_visited_unique": len(set(tuple(p) for p in positions)),
    "token1_first_10": positions[:10].tolist(),
}

# ===== Part 3: numerical stability of additive coupling =====
Xc = X.copy()
vals = []
for t in range(1, 201):
    Xc = permute_sequence(Xc, 1)
    Xc = coupling_mix(Xc, t)
    vals.append(float(np.abs(Xc).max()))
results["coupling_value_growth"] = {"max_value_first": vals[0],
                                    "max_value_t200": vals[-1],
                                    "ratio_per_20_steps":
                                        [round(vals[i] / max(vals[max(0, i - 20)], 1e-12), 3)
                                         for i in range(20, 201, 40)]}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=int)

print("=== State distance d_t (same map applied to both states) ===")
for k, v in state_dist.items():
    a = np.array(v)
    print(f"{k:18s}: mean={a.mean():.5f}  min={a.min():.5f}  max={a.max():.5f}  "
          f"first={v[0]:.5f}  last={v[-1]:.5f}")

print("\n=== Coupling value growth ===")
print("max value t=1:", results["coupling_value_growth"]["max_value_first"],
      "  t=200:", f"{results['coupling_value_growth']['max_value_t200']:.3e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, v in state_dist.items():
        ax[0].plot(range(1, T_MAX + 1), v, label=k, lw=1.3)
    ax[0].set_xlabel("Step t")
    ax[0].set_ylabel("L2 distance d_t")
    ax[0].set_title("State distance under dynamics (same map for both states)")
    ax[0].legend(fontsize=9)

    ax[1].plot(range(1, 201), vals)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Step t")
    ax[1].set_ylabel("max |value| (log)")
    ax[1].set_title("Additive coupling: exponential value growth")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", "state_distance_and_growth.png"), dpi=110)
    plt.close(fig)
    print("plots saved")
except ImportError:
    print("matplotlib not available")
