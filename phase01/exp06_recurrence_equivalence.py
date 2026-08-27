"""exp06_recurrence_equivalence.py — Phase 0.1, Experiment 6.

RECURRENCE + ONE SYSTEM MANY ROUTES + EQUIVALENCE + C_TOTAL.

Part A — ONE MAP, MANY TASKS (recurrence).
  How many distinct functions can a single cat map express by choosing
  different initial states x_0?  The trajectory of x_0 is its ORBIT — a
  cycle of length <= T(N).  The number of DISTINCT orbits is
  ~ N^2 / T(N) ~ O(N).  Each orbit gives one periodic function.
  Compare: N independent blocks give N UNCONSTRAINED functions.
  Parameter saving: 1 map = 4 params vs N blocks x params (weight
  sharing — the standard RNN property, not a chaotic advantage).

Part B — EQUIVALENCE (the decisive negative control of the TZ).
  Route context -> expert:
    chaotic: encoder(context)->x_0, orbit scan -> region
    hash   : context -> expert in O(1)
    trie   : O(len) prefix walk
    MoE    : learned classifier -> expert
  Measured in exp02: orbit covers ~0.75/N of space; random routing
  reachability ~0.5% (single cell, N=128); scan cost O(orbit_len).
  Here we build the C_total table.

Part C — C_TOTAL = construction + training + index + inference + correction.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp06_recurrence_equivalence")
os.makedirs(OUT, exist_ok=True)

A = np.array([[1, 1], [1, 2]], dtype=np.int64)
rng = np.random.default_rng(3)


def period(N):
    a, b, c, d = 1, 0, 0, 1
    t = 0
    while True:
        t += 1
        a, b, c, d = (a + c) % N, (b + d) % N, (a + 2 * c) % N, (b + 2 * d) % N
        if (a, b, c, d) == (1, 0, 0, 1):
            return t


def orbit_len(x0, N, maxl=200000):
    x = x0.copy()
    seen = {}
    for t in range(maxl):
        key = (int(x[0]), int(x[1]))
        if key in seen:
            return t
        seen[key] = t
        x = (A @ x) % N
    return maxl


results = {}

# ============ Part A: expressivity of one map ============
expr = {}
for Ng in [64, 128, 256]:
    T = period(Ng)
    # count distinct orbits by sampling starts and dedup by (length, first cell)
    orbit_ids = set()
    for _ in range(4000):
        x0 = rng.integers(0, Ng, size=2)
        L = orbit_len(x0, Ng)
        # orbit identity: the set of cells; use sorted tuple as key (approx for sample)
        x = x0.copy()
        cells = []
        for _ in range(min(L, 400)):
            cells.append((int(x[0]), int(x[1])))
            x = (A @ x) % Ng
        orbit_ids.add(tuple(sorted(cells)))
    expr[str(Ng)] = {
        "T(N)": T,
        "distinct_orbits_sampled": len(orbit_ids),
        "fraction_of_space_per_orbit": T / (Ng * Ng),
        "min_distinct_orbits_lower_bound": (Ng * Ng) // T,
    }
    print(f"N={Ng:4d}: distinct orbits (sampled 4000 starts) = {len(orbit_ids)}, "
          f"lower bound N^2/T = {(Ng*Ng)//T}")
results["expressivity_one_map"] = expr
results["expressivity_note"] = (
    "One cat map expresses at least N^2/T(N) ~ O(N) distinct orbits "
    "(sampled: 123/245/487 for N=64/128/256), and every orbit is a "
    "LINEAR periodic function (A^t x_0).  N independent blocks express N "
    "UNCONSTRAINED functions.  The parameter saving of one map (4 params "
    "vs N blocks) is exactly RNN weight-sharing — standard, not chaotic — "
    "and it buys only linear periodic routes, far below what N blocks "
    "can express.")

# ============ Part B + C: routing cost table ============
Ng = 128
T = period(Ng)
orbit_mean = 82.1  # measured in exp02
reach_single = 0.0045  # measured in exp02 (single-cell target)
reach_r8 = 0.770  # measured in exp02 (radius-8 ball)

# Chaotic routing (encoder + scan):
#   training: encoder context->x_0   (same class as any learned router)
#   index:    per-orbit cell list (N^2/T orbits x ~T cells = O(N^2) total)
#   inference: orbit scan until region hit; if unreachable, fail/retry
chaotic_scan = orbit_mean  # mean scan length when the target is on the orbit
# with reachability p, expected cost including failures (retry from new x_0):
expected_chaotic_cost = (1.0 / reach_r8) * orbit_mean  # geometric retries
# Note: retries require a NEW x_0 per attempt -> the encoder must emit many
# candidate starts; each failure is a full encoder call + scan.

costs = {
    "chaotic_router": {
        "construction": "encoder training (same as any learned router)",
        "index": f"orbit tables: O(N^2/T * T) = O(N^2) cells (~{(Ng*Ng)//T} orbits x ~{T} cells)",
        "inference": f"orbit scan: O(orbit_len) ~ {orbit_mean} steps; expected with "
                     f"reachability p={reach_r8:.2f}: ~{expected_chaotic_cost:.0f} steps (retries)",
        "correction": "0 (deterministic) or O(T) if controlled",
        "reachability_limit": f"single-cell target reachable only {reach_single:.4f} of the time",
    },
    "hash_table": {
        "construction": "O(K) entries",
        "index": "O(K) memory",
        "inference": "O(1)",
        "correction": "n/a",
        "note": "context->expert is exactly what the chaotic encoder must learn "
                "ANYWAY; the hash does it directly without the orbit scan.",
    },
    "trie": {
        "construction": "O(total_key_len)",
        "index": "O(total_key_len) memory",
        "inference": "O(key_len)",
        "correction": "n/a",
    },
    "moe_router": {
        "construction": "classifier training",
        "index": "O(K) params",
        "inference": "O(K) softmax (or O(log K) with tree)",
        "correction": "n/a",
        "note": "same training cost as the chaotic encoder, no scan, no "
                "reachability limit.",
    },
}
results["routing_cost_table"] = costs
results["C_total"] = {
    "chaotic": "C_train(encoder) + C_index(O(N^2)) + C_infer(O(orbit)) + C_corr(0)"
               " = WORSE than hash on index AND inference, equal on training",
    "hash": "C_index(O(K)) + C_infer(O(1))",
    "moe": "C_train + C_infer(O(K))",
    "verdict": "The chaotic router must train the same encoder as any router, "
               "then ADDS an O(N^2) index and an O(orbit) scan, and is limited "
               "by reachability.  It is a strictly more expensive hash. "
               "(TZ 18.1, 18.2: any win vanishes after preprocessing; chaotic "
               "routing == lookup with extra cost.)",
}
results["hidden_transfer_note"] = (
    "The only 'cheap' part of chaos — the closed form A^t (exp01) — has "
    "zero construction cost, but it matches (not beats) the honest "
    "baseline (matrix power O(log t)).  The O(t) comparison is a "
    "strawman.  Everything else (routing) shifts cost INTO index/training, "
    "exactly the hidden transfer the TZ warns about.")

print("=== Part A: one map expresses O(N) linear periodic functions ===")
for k, v in expr.items():
    print(f"  N={k}: {v['distinct_orbits_sampled']} orbits sampled, "
          f"lower bound {v['min_distinct_orbits_lower_bound']}")
print("\n=== Part B/C: routing cost comparison (N=128) ===")
for k, v in costs.items():
    print(f"  {k:14s}: infer={v['inference']}")
print("\nVerdict: chaotic router = strictly more expensive hash "
      "(same training, +index, +scan, -reachability).")

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
