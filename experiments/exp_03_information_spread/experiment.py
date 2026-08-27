"""exp_03_information_spread/experiment.py

Experiment 0.4 — Information spread.

QUESTIONS
  - Can a single token's signal reach the other tokens under the dynamics?
  - How does the spread depend on the permutation schedule and the coupling?

METHODOLOGY (fair comparison)
  All schedules are used as a FIXED dynamical system: the same map is applied
  every step (this is the correct interpretation of X_{t+1} = F(X_t) for a
  permutation F).  Schedules:
    arnold  : cat map A applied every step
    random  : fresh random permutation every step
    fixed   : one random permutation, repeated
    shift   : cyclic shift by 1 (poor schedule)

  Couplings:
    adj_evenodd : X[2i+1] += 0.5*X[2i]            (one direction)
    adj_sym     : X[2i+1] += 0.5*X[2i]; X[2i] += 0.5*X[2i+1]  (both)

FINDINGS (measured)
  - Pure permutation (no coupling): coverage stays 1/N forever.
    A permutation conserves the support of a state.  (Math fact.)
  - One-directional adjacent coupling: cat map is limited to ~57% of tokens
    (orbit-structure bound); random/fixed reach 100% in ~20-25 steps.
  - Symmetric adjacent coupling: ALL schedules reach 100% in ~log2(N)+2
    steps (support doubles per step: 2^T).  The cat map is no worse than a
    random permutation here — but also no better.  Its advantage is being
    deterministic, parameter-free and exactly reversible.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from chaos_lib import permute_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(123)
T_MAX = 60

results = {}


def run_signal(n_tokens, T, schedule, coupling, rng):
    X = np.zeros((n_tokens, 1))
    X[0, 0] = 1.0
    supports = []
    entropies = []
    l2s = []
    fixed_perm = None
    for t in range(1, T + 1):
        if schedule == "arnold":
            X = permute_sequence(X, 1)
        elif schedule == "random":
            X = X[rng.permutation(n_tokens)]
        elif schedule == "fixed":
            if fixed_perm is None:
                fixed_perm = rng.permutation(n_tokens)
            X = X[fixed_perm]
        elif schedule == "shift":
            X = np.roll(X, 1, axis=0)
        if coupling == "adj_evenodd":
            src = X[0::2].copy()
            X[1::2] = X[1::2] + 0.5 * src
        elif coupling == "adj_sym":
            src = X[0::2].copy()
            X[1::2] = X[1::2] + 0.5 * src
            dst = X[1::2].copy()
            X[0::2] = X[0::2] + 0.5 * dst
        supports.append(int((np.abs(X[:, 0]) > 1e-9).sum()))
        flat = X[:, 0]
        lo, hi = flat.min(), flat.max()
        if hi - lo < 1e-12:
            entropies.append(0.0)
        else:
            hist, _ = np.histogram(flat, bins=16, range=(lo, hi))
            p = hist / hist.sum()
            p = p[p > 0]
            entropies.append(float(-(p * np.log2(p)).sum()))
        l2s.append(float(np.linalg.norm(X)))
    return supports, entropies, l2s


def tth(support, pct, n):
    for i, v in enumerate(support):
        if v >= pct * n:
            return i + 1
    return None


# ---- Fair comparison at N=256 ----
n_tok = 256
rng = RNG
schedules = ["arnold", "random", "fixed", "shift"]
couplings = ["adj_evenodd", "adj_sym"]
for coup in couplings:
    results[f"N256_{coup}"] = {}
    for sched in schedules:
        s, e, l = run_signal(n_tok, T_MAX, sched, coup, rng)
        results[f"N256_{coup}"][sched] = {
            "support": s, "entropy": e, "l2": l,
            "T_25": tth(s, 0.25, n_tok), "T_50": tth(s, 0.5, n_tok),
            "T_90": tth(s, 0.9, n_tok), "T_99": tth(s, 0.99, n_tok),
            "final_coverage": s[-1] / n_tok,
        }

# ---- Scaling: T(99%) vs N for symmetric coupling ----
scaling = {}
for n_tok in [16, 64, 256, 1024, 4096]:
    scaling[str(n_tok)] = {}
    for sched in ["arnold", "random", "fixed", "shift"]:
        s, _, _ = run_signal(n_tok, T_MAX, sched, "adj_sym", rng)
        scaling[str(n_tok)][sched] = {
            "T_99": tth(s, 0.99, n_tok),
            "final_coverage": s[-1] / n_tok,
        }
results["scaling_sym_coupling"] = scaling

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=int)

print("=== N=256, one-directional coupling (adj_evenodd) ===")
for sched in schedules:
    r = results["N256_adj_evenodd"][sched]
    print(f"{sched:8s}: final={r['final_coverage']:.3f}  T25={r['T_25']}  "
          f"T50={r['T_50']}  T90={r['T_90']}  T99={r['T_99']}")

print("\n=== N=256, symmetric coupling (adj_sym) ===")
for sched in schedules:
    r = results["N256_adj_sym"][sched]
    print(f"{sched:8s}: final={r['final_coverage']:.3f}  T25={r['T_25']}  "
          f"T50={r['T_50']}  T90={r['T_90']}  T99={r['T_99']}")

print("\n=== Scaling T(99%) vs N (symmetric coupling) ===")
for n_tok, r in scaling.items():
    row = "  ".join(f"{k}:{v['T_99']}" for k, v in r.items())
    print(f"N={n_tok:>5}: {row}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for sched in schedules:
        r = results["N256_adj_sym"][sched]
        axes[0].plot(range(1, T_MAX + 1), r["support"], label=sched, lw=1.4)
    axes[0].axhline(n_tok, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("Step t")
    axes[0].set_ylabel("Support (nonzero tokens)")
    axes[0].set_title("Symmetric coupling: support growth (N=256)")
    axes[0].legend()

    for sched in schedules:
        r = results["N256_adj_evenodd"][sched]
        axes[1].plot(range(1, T_MAX + 1), r["support"], label=sched, lw=1.4)
    axes[1].axhline(n_tok, color="gray", ls=":", lw=1)
    axes[1].set_xlabel("Step t")
    axes[1].set_ylabel("Support")
    axes[1].set_title("One-directional coupling: support growth (N=256)")
    axes[1].legend()

    ns = [16, 64, 256, 1024, 4096]
    for sched in ["arnold", "random", "fixed", "shift"]:
        ts = [scaling[str(n)][sched]["T_99"] for n in ns]
        axes[2].plot(ns, ts, marker="o", label=sched)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("N (log)")
    axes[2].set_ylabel("T(99%) steps")
    axes[2].set_title("Scaling of spread time vs N (symmetric coupling)")
    axes[2].legend()
    axes[2].plot(ns, [np.log2(n) + 3 for n in ns], "k--", lw=1, label="log2(N)+3")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", "information_spread.png"), dpi=110)
    plt.close(fig)
    print("plots saved")
except ImportError:
    print("matplotlib not available")
