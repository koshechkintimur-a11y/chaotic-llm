"""prepare_public_corpus.py — готовит ПУБЛИЧНЫЙ корпус для воспроизводимого benchmark.

Проблема: phase01/corpus_train.txt — приватный дамп исходников (в .gitignore,
утечка при коммите). Результаты на нём нельзя воспроизвести вне этой машины.

Решение: скачать общедоступный КОРПУС ИСХОДНИКОВ (TheAlgorithms, MIT, pinned
commit), обрезать до того же бюджета символов (MAX_TRAIN=990_000), что и у
приватного прогона, и положить как phase01/corpus_public.txt. Токенизация
(BPE-512, ByteLevel) та же, что в final_benchmark.build_data — меняется только
СОДЕРЖИМОЕ корпуса, протокол идентичен => сравнение честное.

Почему код, а не WikiText/TinyShakespeare: retrieval-тест (A→B на дистанции L)
опирается на точные повторы токенов; в коде идентификаторы/шаблоны повторяются
точно, в худ. тексте — реже. Чтобы ядро «прорыва» (retrieval от хаоса) было
воспроизводимо на публичных данных, берём публичный code-корпус.

Источники (по приоритету, с авто-откатом):
  1. TheAlgorithms/Python @ pinned SHA (MIT) — публичный код, воспроизводим.
  2. TinyShakespeare (karpathy/char-rnn; public domain) — надёжный запасной.
  3. Project Gutenberg #1342 (Pride and Prejudice; public domain) — ещё запасной.

Источник/лицензия/SHA записываются в phase01/corpus_public.SOURCE.md (provenance).

Запуск:
  python prepare_public_corpus.py
  python prepare_public_corpus.py --max-chars 990000 --force
"""
import argparse
import io
import os
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
OUT = os.path.join(PHASE, "corpus_public.txt")
SOURCE_MD = os.path.join(PHASE, "corpus_public.SOURCE.md")
DEFAULT_MAX = 990_000

# Публичный code-корпус: TheAlgorithms (MIT). SHA зашит => воспроизводимость.
CODE_REPOS = [
    ("TheAlgorithms/Python", "9391f546d6f8c72966ff2d7086c28385909bfa5f", "MIT"),
]
SRC_EXT = {".py", ".java", ".go", ".js", ".ts", ".rs", ".cpp", ".c", ".h",
           ".rb", ".sql", ".kt", ".swift", ".scala"}


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (benchmark-corpus-fetch)"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def from_code():
    parts, labels = [], []
    for repo, sha, lic in CODE_REPOS:
        url = f"https://github.com/{repo}/archive/{sha}.tar.gz"
        data = fetch(url)
        z = tarfile.open(fileobj=io.BytesIO(data))
        n = 0
        for m in z.getmembers():
            if m.isfile() and os.path.splitext(m.name)[1] in SRC_EXT:
                try:
                    parts.append(z.extractfile(m).read().decode("utf-8", "ignore"))
                    n += 1
                except Exception:
                    pass
        labels.append(f"{repo}@{sha[:12]} ({lic}, {n} files)")
    return "\n".join(parts), "Public source code: " + "; ".join(labels)


def from_tinyshakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return fetch(url).decode("utf-8", "ignore"), \
        "TinyShakespeare (karpathy/char-rnn; public domain, W. Shakespeare texts)"


def from_gutenberg():
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    return fetch(url).decode("utf-8", "ignore"), \
        "Project Gutenberg #1342 'Pride and Prejudice' (public domain, J. Austen)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX)
    ap.add_argument("--force", action="store_true", help="перезаписать, если уже есть")
    a = ap.parse_args()

    if os.path.exists(OUT) and not a.force:
        print(f"{OUT} уже существует (--force чтобы перезаписать). Выход.")
        return

    sources = [from_code, from_tinyshakespeare, from_gutenberg]
    text, label = None, None
    for fn in sources:
        try:
            text, label = fn()
            print(f"[ok] источник: {label} ({len(text):,} символов)")
            break
        except Exception as e:
            print(f"[skip] {fn.__name__}: {e}")

    if text is None:
        sys.exit("Не удалось скачать ни один публичный корпус.")

    text = text[:a.max_chars]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    with open(SOURCE_MD, "w", encoding="utf-8") as f:
        f.write("# Публичный корпус (corpus_public.txt)\n\n")
        f.write(f"- **Источник:** {label}\n")
        f.write(f"- **Объём:** первые {a.max_chars:,} символов (обрезка для сопоставимости с приватным прогоном)\n")
        f.write("- **Токенизация:** BPE (vocab=512, ByteLevel) — см. `final_benchmark.build_data`\n")
        f.write("- **Лицензия:** см. поле «Источник» выше\n")
        f.write("- **Воспроизведение:** скачать тот же источник (тот же SHA), обрезать до того же\n")
        f.write("  числа символов, запустить `python final_benchmark_v2.py --corpus phase01/corpus_public.txt ...`\n")

    print(f"[done] записано {len(text):,} символов -> {OUT}")
    print(f"[done] provenance -> {SOURCE_MD}")


if __name__ == "__main__":
    main()
