"""rebuild_tables.py — одна команда: adopt + render для README.

Запускать ТОЛЬКО после того, как results/final_benchmark.json перезаписан
реальным прогоном (а не smoke-прогоном со steps=20). Скрипт сам проверяет:
если steps < 1000 — отказывается работать, чтобы не влить smoke-числа в README.

    python rebuild_tables.py            # печатает, что сделает
    python rebuild_tables.py --apply    # реально пишет benchmark/*.json и README

Порядок:
    1. adopt_final_benchmark.py  -> раскладывает results/final_benchmark.json в benchmark/*.json
    2. render_docs.py --inject ../README.md -> вставляет сгенерированные таблицы
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SRC = os.path.join(ROOT, "results", "final_benchmark.json")
MIN_STEPS = 1000  # порог: меньше — считаем smoke/неполным прогоном


def main():
    apply = "--apply" in sys.argv
    if not os.path.exists(SRC):
        sys.exit(f"нет файла {SRC} — прогон ещё не финишировал")
    data = json.load(open(SRC))
    steps = (data.get("_meta") or {}).get("steps")
    n_models = sum(1 for k in data if not k.startswith("_"))
    print(f"источник: {SRC}")
    print(f"  steps={steps}  моделей={n_models}  seeds={data.get('_meta', {}).get('seeds')}")
    if not steps or steps < MIN_STEPS:
        sys.exit(f"ОТКАЗ: steps={steps} < {MIN_STEPS} — похоже на smoke-прогон. "
                 f"Не заливаю эти числа в README. Дождись реального прогона.")

    if not apply:
        print("\n--apply не передан. Ничего не меняю. Для реальной сборки:")
        print("    python rebuild_tables.py --apply")
        return

    import importlib.util

    def run(modname, fnname):
        spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, f"{modname}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getattr(mod, fnname)()

    print("\n[1/2] adopt_final_benchmark.py")
    run("adopt_final_benchmark", "main")
    print("[2/2] render_docs.py --inject ../README.md")
    spec = importlib.util.spec_from_file_location("render_docs", os.path.join(HERE, "render_docs.py"))
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    rd.inject(os.path.join(ROOT, "README.md"))
    print("\nГотово. Теперь можно коммитить benchmark/*.json и README.md.")


if __name__ == "__main__":
    main()
