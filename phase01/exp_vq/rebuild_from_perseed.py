"""rebuild_from_perseed.py — пересборка финального агрегата ИСКЛЮЧИТЕЛЬНО из
персидовых файлов results/bench_v2/*.json (атомарный ground truth, пишутся
харнессом отдельно на каждый сид). Это устраняет любой риск порчи при
параллельной записи двух процессов в один final_benchmark_v2.json.

_Не_ пересчитывает ничего — только собирает уже готовые сиды и переносит
_meta / _scaling из текущего агрегата (они корректны и общие для обоих прогонов).
"""
import json, re, os, glob, math, statistics, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
BENCH = os.path.join(ROOT, "results", "bench_v2")
AGG = os.path.join(ROOT, "results", "final_benchmark_v2.json")
BACKUP = AGG + ".prerebuild_backup"

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_crit(n):
    return _T95.get(n, 1.96)


def ci95(vals):
    n = len(vals)
    if n < 2:
        return None, None
    m = statistics.mean(vals)
    s = statistics.pstdev(vals)
    se = s / math.sqrt(n)
    return m, t_crit(n) * se


# (имя конфига в агрегате) -> (regex персидового файла, ожидаемые params)
CONFIGS = {
    "STS-Prog":                   (r"sts_prog_seed(\d+)\.json$", 900353),
    "STS-Prog (no-PC)":           (r"sts_prog__no_pc__seed(\d+)\.json$", 900353),
    "Transformer (D=88)":         (r"transformer__d_88__seed(\d+)\.json$", 865904),
    "Transformer-matched (D=92)": (r"transformer_matched__d_92__seed(\d+)\.json$", 940568),
}

all_results = {}
for name, (pat, exp_params) in CONFIGS.items():
    rx = re.compile(pat)
    entries = []
    for fp in glob.glob(os.path.join(BENCH, "*.json")):
        m = rx.match(os.path.basename(fp))
        if not m:
            continue
        seed = int(m.group(1))
        e = json.load(open(fp, encoding="utf-8"))
        assert e["params"] == exp_params, f"{name} seed {seed}: params {e['params']} != {exp_params}"
        assert e["seed"] == seed, f"{name}: seed {seed} != file seed {e['seed']}"
        assert "retrieval" in e and "train" in e and "ppl" in e
        entries.append((seed, e))
    entries.sort(key=lambda x: x[0])
    seeds = [e for _, e in entries]
    ppls = [e["ppl"] for e in seeds]
    m_mean, m_ci = ci95(ppls)
    all_results[name] = {
        "seeds": seeds,
        "stats": {
            "mean_ppl": round(m_mean, 3) if m_mean is not None else None,
            "ppl_95ci": [round(m_mean - m_ci, 3), round(m_mean + m_ci, 3)] if m_ci is not None else None,
            "std_ppl": round(statistics.pstdev(ppls), 3) if len(ppls) > 1 else 0.0,
            "n_seeds": len(ppls),
        },
    }
    print(f"{name:32s} n={len(seeds)} mean_ppl={m_mean:.3f} "
          f"ci95={all_results[name]['stats']['ppl_95ci']} std={all_results[name]['stats']['std_ppl']}")

# переносим _meta и _scaling из текущего агрегата (корректны, общие для обоих прогонов)
if os.path.exists(AGG):
    old = json.load(open(AGG, encoding="utf-8"))
    for k in ("_meta", "_scaling"):
        if k in old:
            all_results[k] = old[k]
            print("carried over", k)

# бэкап текущего, затем запись чистого агрегата
if os.path.exists(AGG):
    shutil.copy2(AGG, BACKUP)
    print("backed up ->", os.path.basename(BACKUP))
with open(AGG, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("WROTE", AGG)
