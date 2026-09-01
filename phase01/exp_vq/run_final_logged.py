"""run_final_logged.py — обёртка над final_benchmark.py, которая НЕ теряет stdout.

Проблема: final_benchmark.py пишет JSON только один раз, в самом конце,
а его stdout никуда не перенаправляется. Если процесс умрёт на 8-м прогоне
из 9 — не останется ни результатов, ни строчки лога.

Эта обёртка:
  1. зеркалит весь stdout+stderr в results/final_benchmark_<timestamp>.log;
  2. сохраняет PID, время старта и команду в results/final_benchmark_<timestamp>.meta.json;
  3. НИЧЕГО не меняет в final_benchmark.py (тот же интерпретатор, тот же cwd).

Запуск:
    cd phase01/exp_vq
    python run_final_logged.py                 # запустить final_benchmark.py
    python run_final_logged.py -- other.py     # или любой другой скрипт
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    target = argv[0] if argv else "final_benchmark.py"
    target_args = argv[1:]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(RESULTS, f"final_benchmark_{stamp}.log")
    meta_path = os.path.join(RESULTS, f"final_benchmark_{stamp}.meta.json")

    cmd = [sys.executable, "-u", target] + target_args
    meta = {
        "cmd": cmd,
        "cwd": HERE,
        "python": sys.executable,
        "started": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"[run_final_logged] лог:  {log_path}")
    print(f"[run_final_logged] ком.: {' '.join(cmd)}")
    sys.stdout.flush()

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        meta["pid"] = proc.pid
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
        except KeyboardInterrupt:
            proc.terminate()
            print("\n[run_final_logged] прервано пользователем", file=sys.stderr)
        rc = proc.wait()
    elapsed = time.time() - t0

    meta["exit_code"] = rc
    meta["elapsed_s"] = round(elapsed, 1)
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    print(f"\n[run_final_logged] exit={rc} за {elapsed/60:.1f} мин")
    print(f"[run_final_logged] лог: {log_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
