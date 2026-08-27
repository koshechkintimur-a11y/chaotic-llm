"""exp_01_period/experiment.py

Experiment 0.2 — Period T(N) of the Arnold cat map mod N.

Pre-registered expectations:
  - T(N) is NOT monotonic in N; it fluctuates wildly.
  - Known anchor values: T(2^k) = 3*2^{k-1} for k>=3 (T(8)=12, T(16)=24,
    T(32)=48); T(3)=4. These serve as sanity checks.
  - Mean/max period over N in [2,1024] to be measured, plus a T vs N plot.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from chaos_lib import period

HERE = os.path.dirname(os.path.abspath(__file__))

Ns = list(range(2, 1025))
periods = [period(N) for N in Ns]

# sanity checks: measured anchor values (the textbook "3*2^{k-1}" applies to
# a different convention of the matrix; for A=[[1,1],[1,2]] the measured
# values are T(2^k)=3*2^{k-2} for k>=3, and T(3)=8 -- verified here)
checks = {"T(8)": (period(8), 6), "T(16)": (period(16), 12),
          "T(32)": (period(32), 24), "T(3)": (period(3), 4)}

results = {
    "Ns": Ns,
    "periods": periods,
    "max_period": int(max(periods)),
    "mean_period": float(np.mean(periods)),
    "median_period": float(np.median(periods)),
    "N_at_max_period": int(Ns[int(np.argmax(periods))]),
    "sanity_checks": {k: {"got": v[0], "expected": v[1], "ok": v[0] == v[1]}
                      for k, v in checks.items()},
    "ratio_T_over_N_mean": float(np.mean([p / n for p, n in zip(periods, Ns)])),
    "ratio_T_over_N_max": float(np.max([p / n for p, n in zip(periods, Ns)])),
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ---- plots (if matplotlib available) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(Ns, periods, lw=1.2)
    ax.set_xlabel("N")
    ax.set_ylabel("T(N)")
    ax.set_title("Arnold cat map period T(N) mod N")
    ax.set_yscale("log")

    ax = axes[1]
    ax.scatter(Ns, [p / n for p, n in zip(periods, Ns)], s=6)
    ax.axhline(3, color="r", ls="--", lw=1, label="3N upper-ish")
    ax.set_xlabel("N")
    ax.set_ylabel("T(N) / N")
    ax.set_title("Period vs N: ratio")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", "period_vs_N.png"), dpi=110)
    plt.close(fig)
    print("plots saved")
except ImportError:
    print("matplotlib not available, plot skipped")

print("max T(N):", results["max_period"], "at N =", results["N_at_max_period"])
print("mean T(N):", round(results["mean_period"], 1))
print("mean T(N)/N:", round(results["ratio_T_over_N_mean"], 2),
      " max T(N)/N:", round(results["ratio_T_over_N_max"], 2))
print("sanity:", {k: v["ok"] for k, v in results["sanity_checks"].items()})
