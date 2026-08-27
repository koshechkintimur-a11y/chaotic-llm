"""exp_00_reversibility/experiment.py

Experiment 0.1 — Reversibility of the Arnold cat map on Z_N^2.

Pre-registered expectation:
  - det(A)=1 and the map is a bijection on Z_N^2, so F^{-1}(F(X)) = X
    must hold EXACTLY (not approximately) for every N and every state.
  - On a finite space every state lies on a cycle; we measure orbit periods.
  - The token-sequence interpretation (permute_sequence) must also be
    exactly reversible, including the padding convention.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from chaos_lib import (arnold_map, arnold_inv, period, permute_indices,
                       permute_sequence)

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [8, 16, 32, 64, 128, 256, 512, 1024]
ITERS = [1, 2, 3, 7, 11]
RNG = np.random.default_rng(12345)

results = {"Ns": NS, "iters": ITERS, "runs": {}}

for N in NS:
    entry = {}
    n_states = min(2000, N * N)
    states = RNG.integers(0, N, size=(n_states, 2))
    worst_err = 0
    for t in ITERS:
        fwd = arnold_map(states, N, t)
        back = arnold_inv(fwd, N, t)
        err = int(np.abs(back - states).max())
        worst_err = max(worst_err, err)
        entry[f"inv_t{t}_maxerr"] = err
        entry[f"inv_t{t}_exact_frac"] = float(np.all(back == states, axis=-1).mean())
    # orbit periods of individual states (cycle structure)
    periods = set()
    sample = states[:min(200, N * N)]
    for s in sample:
        cur = s.copy()
        seen = {}
        for step in range(1, 20000):
            cur = arnold_map(cur.reshape(1, 2), N, 1).reshape(2)
            key = (int(cur[0]), int(cur[1]))
            if key in seen:
                periods.add(step - seen[key])
                break
            seen[key] = step
    entry["state_orbit_periods_sample"] = sorted(periods)
    entry["state_orbit_max"] = max(periods, default=0)
    entry["map_period_T(N)"] = period(N)
    entry["worst_inv_error"] = worst_err
    results["runs"][str(N)] = entry

# token-sequence reversibility (the object we actually use downstream)
# NOTE: requires n_tokens to be a perfect square (mathematical constraint)
seq_results = {}
for n_tok in [16, 64, 256]:
    X = RNG.normal(size=(n_tok, 4))
    for t in [1, 2, 5, 9]:
        Y = permute_sequence(X, t)
        Yb = permute_sequence(Y, -t)  # undo via the inverse map
        ok = np.allclose(Yb, X)
        seq_results[f"tok{n_tok}_t{t}"] = bool(ok)
    # composition of several forward steps must equal one step of total t
    X2 = X.copy()
    for t in [1, 2, 5]:
        X2 = permute_sequence(X2, t)
    seq_results[f"tok{n_tok}_compose_1_2_5"] = bool(
        np.allclose(X2, permute_sequence(X, 1 + 2 + 5)))
results["token_sequence_reversibility"] = seq_results

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=int)

print("max inversion error across all N, t:", max(
    r["worst_inv_error"] for r in results["runs"].values()))
print("token sequence reversibility all ok:", all(seq_results.values()))
print("sample periods per N:")
for N in NS:
    print(f"  N={N:4d}  T(N)={results['runs'][str(N)]['map_period_T(N)']:5d}  "
          f"orbit max (sample)={results['runs'][str(N)]['state_orbit_max']}")
