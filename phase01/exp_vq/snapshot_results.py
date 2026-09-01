"""snapshot_results.py — сторож: снимает копию results_*.json при каждом изменении.

Зачем: experiment_pc.py пишет результат в results_{config}.json, а config для всех
режимов равен "pc". Поэтому несколько прогонов подряд перезаписывают один файл,
и число предыдущего конфига исчезает бесследно (так пропали 19.22 / 47%).

Сторож ничего не меняет в работе запущенного прогона — только читает файлы
и кладёт копии в _snapshots/. Копия снимается только если файл парсится как
JSON, иначе можно снять недописанный файл.

Смотрит за двумя местами (второе добавлено после того, как выяснилось, что
final_benchmark.py пишет НЕ в exp_vq/, а в chaotic-llm/results/):
    exp_vq/results_*.json   — старые прогоны experiment_pc.py
    ../../results/*.json    — final_benchmark.py и компания

    python snapshot_results.py            # один проход
    python snapshot_results.py --watch 10 # каждые 10 секунд, пока не убьёшь
"""

import argparse
import hashlib
import json
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "_snapshots")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# (каталог, префикс имени) — откуда снимать копии
WATCH = [
    (HERE, "results_"),   # exp_vq/results_*.json
    (os.path.join(REPO, "results"), ""),  # chaotic-llm/results/*.json (все)
]


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def valid_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def sweep(state, quiet=False, copy=True):
    changed = []
    for watch_dir, prefix in WATCH:
        if not os.path.isdir(watch_dir):
            continue
        for fn in sorted(os.listdir(watch_dir)):
            if not (fn.startswith(prefix) and fn.endswith(".json")):
                continue
            path = os.path.join(watch_dir, fn)
            if not os.path.isfile(path):
                continue
            try:
                d = digest(path)
            except OSError:
                continue
            if state.get(path) == d:
                continue
            if not valid_json(path):
                continue  # файл дописывается — возьмём на следующем проходе
            known = state.get(path)
            state[path] = d
            if known is None or not copy:
                continue  # первичный проход: только запоминаем, ничего не копируем
            os.makedirs(SNAP, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            sub = "" if watch_dir == HERE else "repo-results_"
            dst = os.path.join(SNAP, f"{sub}{fn[:-5]}-{stamp}.json")
            shutil.copy2(path, dst)
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                tag = rec.get("config") or rec.get("driver_mode") or rec.get("kind") or "?"
                brief = f"mixer={rec.get('mixer_ppl')} retr={rec.get('retrieval')}"
                if brief == "mixer=None retr=None":
                    keys = [k for k in rec if not k.startswith("_")]
                    brief = f"секций={len(keys)} шаги={rec.get('_meta', {}).get('steps')}"
            except Exception:
                tag, brief = "?", ""
            changed.append((fn, dst, tag, brief))
            if not quiet:
                print(f"[{stamp}] снимок {os.path.relpath(path, REPO)} "
                      f"-> {os.path.basename(dst)}  ({tag}) {brief}", flush=True)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0.0,
                    help="интервал повтора в секундах; 0 = один проход")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    state = {}
    sweep(state, quiet=True, copy=False)  # запоминаем, что уже лежит, без копий
    if args.watch <= 0:
        sweep(state, quiet=args.quiet)
        return
    print(f"сторож запущен, интервал {args.watch:g} с, снимки в {SNAP}", flush=True)
    try:
        while True:
            sweep(state, quiet=args.quiet)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("сторож остановлен", flush=True)


if __name__ == "__main__":
    main()
