"""run_night_chain.py — ночная цепочка scale-up (пропускает готовый 50M chaos).

Runs sequentially: 50M tf -> 100M chaos -> 100M tf (each 8000 steps).
50M chaos already done (scale_50M_chaos.json on disk) — skipped.
Writes scale_summary.json + SCALE_REPORT.md verdict at the end.
"""
import os
import sys
import json
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUNS = [
    (50, "tf"),
    (100, "chaos"),
    (100, "tf"),
]
STEPS = 8000


def main():
    results = []
    # keep existing 50M chaos result
    if os.path.exists(os.path.join(HERE, "scale_50M_chaos.json")):
        c = json.load(open(os.path.join(HERE, "scale_50M_chaos.json")))
        results.append({"budget": 50, "kind": "chaos", "exit": 0, "time_s": None,
                        "mixer_only": c["mixer_only"], "gated": c["gated"],
                        "params": c["params"]})
        print("50M chaos: REUSED from disk (PPL=%.2f gated=%.2f)" % (
            c["mixer_only"], c["gated"]["1.0"]), flush=True)
    t0 = time.time()
    for budget, kind in RUNS:
        print(f"\n===== START {budget}M {kind} ({STEPS} steps) =====", flush=True)
        t1 = time.time()
        cmd = [sys.executable, "scale_train.py", str(budget), kind, str(STEPS)]
        r = subprocess.run(cmd, cwd=HERE, capture_output=False)
        dt = time.time() - t1
        jf = os.path.join(HERE, f"scale_{budget}M_{kind}.json")
        res = None
        if os.path.exists(jf):
            res = json.load(open(jf))
        results.append({
            "budget": budget, "kind": kind, "exit": r.returncode,
            "time_s": round(dt, 1),
            "mixer_only": res.get("mixer_only") if res else None,
            "gated": res.get("gated") if res else None,
            "params": res.get("params") if res else None,
        })
        print(f"===== DONE {budget}M {kind} exit={r.returncode} time={dt:.0f}s =====", flush=True)

    summary = {"total_time_s": round(time.time() - t0, 1), "runs": results}
    json.dump(summary, open(os.path.join(HERE, "scale_summary.json"), "w"), indent=2)
    print("\n##### NIGHT CHAIN COMPLETE #####", flush=True)
    for r in results:
        print(f"  {r['budget']}M {r['kind']}: PPL={r['mixer_only']} gated={r['gated']}", flush=True)
    print("saved scale_summary.json", flush=True)


if __name__ == "__main__":
    main()
