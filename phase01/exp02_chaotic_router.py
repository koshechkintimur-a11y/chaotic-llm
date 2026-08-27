"""exp02_chaotic_router.py — Phase 0.1, Experiment 2.

CHAOS AS ADDRESS SPACE / ROUTER.

Question: can the chaotic trajectory serve as a route from an encoded
initial state x_0 to a target region (expert), cheaper than a baseline
router (hash / MoE / scan)?

KEY STRUCTURAL FACT to measure (pre-registered):
  The cat map is a PERMUTATION on Z_N^2 with period T(N) ~ 0.75N (powers
  of two).  Therefore the orbit of ANY x_0 visits at most T(N) distinct
  cells — a fraction T(N)/N^2 ~ 0.75/N of the whole space.  For N=128:
  96/16384 = 0.6%.  A router whose reachable set from one x_0 is 0.6% of
  the space cannot reach most targets.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp02_chaotic_router")
os.makedirs(OUT, exist_ok=True)

A = np.array([[1, 1], [1, 2]], dtype=np.int64)


def step(x, N):
    return (A @ x) % N


def orbit(x0, N, max_len=200000):
    """Return the list of cells visited before repetition."""
    seen = {}
    x = x0.copy()
    path = []
    for t in range(max_len):
        key = (int(x[0]), int(x[1]))
        if key in seen:
            return path, seen[key]
        seen[key] = t
        path.append(key)
        x = step(x, N)
    return path, None


def period(N):
    a, b, c, d = 1, 0, 0, 1
    t = 0
    while True:
        t += 1
        a, b, c, d = (a + c) % N, (b + d) % N, (a + 2 * c) % N, (b + 2 * d) % N
        if (a, b, c, d) == (1, 0, 0, 1):
            return t


results = {}

# ============ Part 1: orbit structure ============
rng = np.random.default_rng(0)
orbits_data = {}
for Ng in [32, 64, 128, 256]:
    T = period(Ng)
    lens = []
    fracs = []
    for _ in range(200):
        x0 = rng.integers(0, Ng, size=2)
        path, _ = orbit(x0, Ng)
        lens.append(len(path))
        fracs.append(len(path) / (Ng * Ng))
    orbits_data[str(Ng)] = {
        "T(N)": T,
        "orbit_len_mean": float(np.mean(lens)),
        "orbit_len_max_sample": int(np.max(lens)),
        "fraction_of_space_mean": float(np.mean(fracs)),
        "fraction_of_space_max": float(np.max(fracs)),
    }
    print(f"N={Ng:4d}: T(N)={T:5d}  orbit_len mean={np.mean(lens):7.1f} "
          f"frac of space mean={np.mean(fracs):.4f}")
results["orbit_structure"] = orbits_data

# ============ Part 2: routing success ============
# Can a random x_0 reach a random target region?
# Region = single random cell (the hardest, cleanest case).
Ng = 128
n_trials = 2000
reachable = 0
hit_times = []
for _ in range(n_trials):
    x0 = rng.integers(0, Ng, size=2)
    target = tuple(rng.integers(0, Ng, size=2))
    path, _ = orbit(x0, Ng)
    if target in path:
        reachable += 1
        hit_times.append(path.index(target))
results["routing_single_cell"] = {
    "N": Ng,
    "space_cells": Ng * Ng,
    "target_reachable_frac_random_x0": reachable / n_trials,
    "mean_hit_time_if_reachable": float(np.mean(hit_times)) if hit_times else None,
    "expected_hit_time_naive_scan": Ng * Ng / 2,  # scan half the grid
}
print(f"\nRouting to a random single cell: reachable from random x_0 in "
      f"{reachable/n_trials:.4f} of trials (orbit covers ~{96/16384:.4f} of space)")

# ============ Part 3: routing with larger regions ============
# Regions of radius r (balls).  Reachability grows with region size S.
for r in [1, 3, 5, 8]:
    reachable = 0
    n_trials = 1000
    for _ in range(n_trials):
        x0 = rng.integers(0, Ng, size=2)
        c = rng.integers(r, Ng - r, size=2)
        path, _ = orbit(x0, Ng)
        for cell in path:
            if max(abs(cell[0] - c[0]), abs(cell[1] - c[1])) <= r:
                reachable += 1
                break
    frac = reachable / n_trials
    print(f"  region radius r={r:2d} (S~{(2*r+1)**2:5d} cells, {((2*r+1)**2)/(Ng*Ng):.2%} of space): "
          f"reachable {frac:.3f}")
    results[f"routing_ball_r{r}"] = {
        "region_size": (2 * r + 1) ** 2,
        "reachable_frac": frac,
    }

# ============ Part 4: the "addressable jump" vs "finding t" ============
# Jumping to A^t x_0 is O(log t).  But FINDING the t that hits a region is
# a search — measured as scan cost above.  The addressable jump helps only
# if you already know t (e.g., region = orbit position t for a known t),
# which requires the encoder to know the region = it IS the router.
results["addressable_vs_search_note"] = (
    "Jumping to state at step t costs O(log t) (exp01).  But 'find the t "
    "that hits region R' is a search over the orbit: O(orbit_len) scan or "
    "an O(orbit_len) precomputed index.  The addressable jump does NOT "
    "remove the search.  The measured reachable fraction (Part 2/3) is "
    "the fundamental limit: from a random x_0 the orbit covers ~0.6% of "
    "the space (N=128), so most targets are unreachable.")

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
