"""adopt_final_benchmark.py — переносит results/final_benchmark.json в benchmark/*.json.

final_benchmark.py пишет всё одним файлом в конце прогона. Это удобно для чтения,
но это же единственная точка отказа: упадёт процесс — пропадут все девять
прогонов. Здесь результаты разбираются в схему benchmark/ (по файлу на прогон),
после чего render_docs.py собирает из них таблицы README.

    python adopt_final_benchmark.py
    python render_docs.py --inject ../README.md
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SRC = os.path.join(ROOT, "results", "final_benchmark.json")
BENCH = os.path.join(HERE, "benchmark")

SLUG = {
    "STS-Prog": "sts_prog_final",
    "STS-Prog (no-PC)": "nopc_final",
    "Transformer (D=88)": "transformer_final",
}

DRIVER = {
    "STS-Prog": "sts_prog",
    "STS-Prog (no-PC)": "sts_prog_nopc",
    "Transformer (D=88)": "__transformer__",
}

# Данные, которых в final_benchmark.json нет, но которые нужны для таблицы.
# Не выдумываем: если значение неизвестно, пишем None и таблица покажет прочерк.
HARDCODED_FACTS = {
    "sts_prog_final": dict(d=192, layers=8, model="PurePCLM"),
    "nopc_final": dict(d=192, layers=8, model="PurePCLM"),
    "transformer_final": dict(d=88, layers=8, model="TransformerLM"),
}


def adopt_retrieval(ret):
    out = {}
    for L, r in (ret or {}).items():
        key = f"L{L}"
        out[key] = {
            "hits": r.get("hits"),
            "n_valid": r.get("trials"),
            "rate": r.get("accuracy"),
        }
    return out


def main():
    if not os.path.exists(SRC):
        sys.exit(f"нет файла {SRC} — прогон ещё не финишировал")
    with open(SRC) as f:
        data = json.load(f)

    meta = data.get("_meta", {})
    os.makedirs(BENCH, exist_ok=True)
    written = []

    for name, payload in data.items():
        if name.startswith("_"):
            continue
        slug = SLUG.get(name)
        if slug is None:
            slug = "".join(c if c.isalnum() else "_" for c in name).lower()
        facts = HARDCODED_FACTS.get(slug, {})

        for seed_entry in payload.get("seeds", []):
            seed = seed_entry.get("seed")
            rec = {
                "name": f"{slug}_seed{seed}",
                "driver_mode": DRIVER.get(name, name),
                "model": facts.get("model"),
                "d": facts.get("d"),
                "layers": facts.get("layers"),
                "vocab": None,
                "params": seed_entry.get("params"),
                "results": {
                    "mixer_ppl": seed_entry.get("ppl"),
                    "gated_ppl": None,  # в final_benchmark не считается
                    "n_eval": None,
                },
                "retrieval": adopt_retrieval(seed_entry.get("retrieval")),
                "provenance": {
                    "steps": meta.get("steps"),
                    "batch": meta.get("batch"),
                    "lr": meta.get("lr"),
                    "warmup": meta.get("warmup"),
                    "seed": seed,
                    "seed_retrieval": seed,
                    "time_s": (seed_entry.get("train") or {}).get("time"),
                    "source": os.path.relpath(SRC, ROOT),
                    "git": {"commit": meta.get("commit"), "dirty": None, "branch": None},
                    "env": {"device": "cuda"},
                    "corpus": {"name": "последние 20% train-корпуса", "path": None,
                               "sha256_8mb": None, "n_train_tokens": None,
                               "n_test_tokens": None, "vocab": None},
                },
            }
            path = os.path.join(BENCH, f"{slug}_seed{seed}.json")
            with open(path, "w") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            written.append(os.path.basename(path))

        stats = payload.get("stats") or {}
        agg = {
            "name": f"{slug}_agg",
            "driver_mode": DRIVER.get(name, name),
            "model": facts.get("model"),
            "d": facts.get("d"),
            "layers": facts.get("layers"),
            "params": (payload.get("seeds") or [{}])[0].get("params"),
            "n_seeds": len(payload.get("seeds", [])),
            "results": {
                "mixer_ppl_mean": stats.get("mean_ppl"),
                "mixer_ppl_std": stats.get("std_ppl"),
                "mixer_ppl_min": stats.get("min_ppl"),
                "mixer_ppl_max": stats.get("max_ppl"),
            },
            "retrieval_mean": {},
        }
        dists = {}
        for seed_entry in payload.get("seeds", []):
            for L, r in (seed_entry.get("retrieval") or {}).items():
                dists.setdefault(f"L{L}", []).append(r.get("accuracy"))
        for L, vals in dists.items():
            agg["retrieval_mean"][L] = {
                "mean": round(sum(vals) / len(vals), 4),
                "std": round(float(__import__("statistics").pstdev(vals)) if len(vals) > 1 else 0.0, 4),
                "n_seeds": len(vals),
            }
        path = os.path.join(BENCH, f"{slug}_agg.json")
        with open(path, "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        written.append(os.path.basename(path))

    scaling = data.get("_scaling")
    if scaling:
        path = os.path.join(BENCH, "scaling_final.json")
        with open(path, "w") as f:
            json.dump({"name": "scaling_final", "scaling": scaling,
                       "note": "один прямой проход, batch=1, модель не обучена, без прогрева"},
                      f, indent=2, ensure_ascii=False)
        written.append(os.path.basename(path))

    print(f"записано {len(written)} файлов в {BENCH}:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
