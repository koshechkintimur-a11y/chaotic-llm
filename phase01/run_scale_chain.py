"""run_scale_chain.py — automatic sequential 4-run chain for 50M/100M scale-up.

Runs: 50 chaos, 50 tf, 100 chaos, 100 tf (each 8000 steps).
Writes scale_summary.json at the end. Notifies on completion.
"""
import os
import sys
import json
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUNS = [
    (50, "chaos"),
    (50, "tf"),
    (100, "chaos"),
    (100, "tf"),
]
STEPS = 8000


def main():
    results = []
    t0 = time.time()
    for budget, kind in RUNS:
        print(f"\n===== START {budget}M {kind} ({STEPS} steps) =====", flush=True)
        t1 = time.time()
        cmd = [sys.executable, "scale_train.py", str(budget), kind, str(STEPS)]
        r = subprocess.run(cmd, cwd=HERE, capture_output=False)
        dt = time.time() - t1
        # load result json
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
    summary = {
        "total_time_s": round(time.time() - t0, 1),
        "runs": results,
    }
    json.dump(summary, open(os.path.join(HERE, "scale_summary.json"), "w"), indent=2)
    print("\n##### SCALE CHAIN COMPLETE #####", flush=True)
    for r in results:
        print(f"  {r['budget']}M {r['kind']}: PPL={r['mixer_only']} gated={r['gated']}", flush=True)
    print("saved scale_summary.json", flush=True)


if __name__ == "__main__":
    main()
