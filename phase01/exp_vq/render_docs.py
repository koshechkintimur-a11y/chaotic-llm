"""render_docs.py — собирает таблицы README из benchmark/*.json.

Правило: в README попадает только то, что лежит в файле. Если файла нет,
в таблице стоит «нет артефакта», а не число из памяти или из сообщения коммита.
Это убирает класс ошибок, из-за которого 19.22 / 47% существовали только в тексте.

Источник данных — файлы, которые пишет adopt_final_benchmark.py из
results/final_benchmark.json (по одному на сид: <slug>_seed{seed}.json).

    python render_docs.py                 # печатает таблицы в stdout
    python render_docs.py --inject ../README.md
    python render_docs.py --bench /tmp/x  # для самопроверки
"""

import argparse
import json
import os
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benchmark")
BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"

# Порядок дистанций в таблице retrieval (совпадает с RETRIEVAL_L в final_benchmark.py).
DISTS = ("L16", "L32", "L64", "L128", "L256")

# (отображаемая метка, slug-префикс файлов в benchmark/)
MODELS = [
    ("STS-Prog", "sts_prog_final"),
    ("STS-Prog (без хаоса, nopc)", "nopc_final"),
    ("Transformer (D=88)", "transformer_final"),
]


def load_seeds(slug, bench):
    files = sorted(glob.glob(os.path.join(bench, f"{slug}_seed*.json")))
    recs = []
    for p in files:
        try:
            with open(p) as f:
                recs.append(json.load(f))
        except Exception:
            pass
    return recs


def fmt(v, nd=3):
    if v is None:
        return "нет артефакта"
    return f"{v:.{nd}f}"


def pool_retrieval(recs):
    """Пулит retrieval по всем сидам: суммирует hits / n_valid -> честный знаменатель."""
    out = {}
    for L in DISTS:
        hits = 0
        n = 0
        for r in recs:
            cell = (r.get("retrieval") or {}).get(L)
            if cell is None:
                continue
            hits += cell.get("hits") or 0
            n += cell.get("n_valid") or 0
        out[L] = (hits / n, n) if n > 0 else (None, 0)
    return out


def ppl_stats(recs):
    vals = [r["results"].get("mixer_ppl") for r in recs
            if r.get("results") and r["results"].get("mixer_ppl") is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std


def ret_cell(pooled, L):
    rate, n = pooled.get(L, (None, 0))
    if rate is None:
        return "нет артефакта"
    return f"{rate*100:.1f}% (n={n})"


def row_for(model, slug, bench):
    recs = load_seeds(slug, bench)
    if not recs:
        cells = " | ".join(["нет артефакта"] * (len(DISTS) + 2))
        return f"| {model} | {cells} |"
    mean, std = ppl_stats(recs)
    pooled = pool_retrieval(recs)
    params = recs[0].get("params")
    ppl = f"{mean:.3f}±{std:.3f}" if mean is not None else "нет артефакта"
    return (f"| {model} | {ppl} | {params:,} | "
            + " | ".join(ret_cell(pooled, L) for L in DISTS) + " |")


def build(bench=BENCH):
    out = []
    out.append("\n### Параметр-матч benchmark (`final_benchmark.py`, 8 слоёв)\n")
    header = "| Модель | PPL (mean±std) | Параметры | " + \
             " | ".join(f"Retr {d[1:]}" for d in DISTS) + " |"
    out.append(header)
    out.append("|" + "---|" * (len(DISTS) + 3))
    for model, slug in MODELS:
        out.append(row_for(model, slug, bench))

    # Провенанс — берём из первого попавшегося файла любой модели.
    prov = {}
    for _, slug in MODELS:
        recs = load_seeds(slug, bench)
        if recs:
            prov = recs[0].get("provenance", {})
            break

    out.append("\n### Откуда числа\n")
    out.append("| Модель | Файл | Коммит | Шаги | Batch | LR | Warmup | Устройство |")
    out.append("|" + "---|" * 8)
    for model, slug in MODELS:
        recs = load_seeds(slug, bench)
        if not recs:
            out.append(f"| {model} | `benchmark/{slug}_seedN.json` | — | — | — | — | — | — |")
            continue
        p = recs[0].get("provenance", {})
        g = p.get("git", {}) or {}
        e = p.get("env", {}) or {}
        commit = (g.get("commit") or "?")[:8]
        if g.get("dirty"):
            commit += " (грязный)"
        out.append(f"| {model} | `benchmark/{slug}_seed0.json` | `{commit}` | "
                   f"{p.get('steps')} | {p.get('batch')} | {p.get('lr')} | "
                   f"{p.get('warmup')} | {e.get('device', e.get('gpu'))} |")

    out.append("\n> Retrieval = доля правильных предсказаний B в паттерне A→B, "
               "усреднено по **всем принятым пробам** (n_trials=200 на сид, пул по 3 сидам). "
               "Знаменатель записан в `benchmark/*.json`. PPL считается на последних 20% "
               "train-корпуса (`eval_ppl`, seed=42).")
    return "\n".join(out) + "\n"


def inject(path, bench=BENCH):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    block = f"{BEGIN}\n{build(bench)}\n{END}"
    if BEGIN in text and END in text:
        before, rest = text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"таблицы вставлены в {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=str, default="")
    ap.add_argument("--bench", type=str, default=BENCH)
    a = ap.parse_args()
    if a.inject:
        inject(a.inject, a.bench)
    else:
        print(build(a.bench))
