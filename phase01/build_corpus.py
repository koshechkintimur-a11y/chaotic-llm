"""build_corpus.py — collect the user's code projects into one corpus file.

Mimics morin-filter experiment W: real code from the user's projects.
Split 4:1 BY FILE (honest, no context leak).
"""
import os
import random

ROOT = r"C:\Users\Geroin\Desktop\03_Проекты"
EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".py"}
EXCLUDE_DIRS = {"node_modules", "dist", "build", ".git", "__pycache__",
                "venv", ".venv", "target", ".next", "coverage", "swiftshader",
                "chroma_db", "disk3", "v2rayN-windows-64", "nekoray",
                "OpenDevin.OpenDevin-main", "discord", "bin"}
MIN_SIZE = 1000

files = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in EXTENSIONS:
            continue
        p = os.path.join(dirpath, fn)
        try:
            if os.path.getsize(p) >= MIN_SIZE:
                files.append(p)
        except OSError:
            pass

random.seed(42)
random.shuffle(files)
n = len(files)
n_train = int(n * 0.8)
train_files, test_files = files[:n_train], files[n_train:]

def dump(paths, out_path):
    total_chars = 0
    with open(out_path, "w", encoding="utf-8", errors="ignore") as out:
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            out.write(f"\n// FILE: {os.path.basename(p)}\n")
            out.write(text)
            out.write("\n")
            total_chars += len(text)
    return total_chars

c_train = dump(train_files, r"C:\Users\Geroin\chaotic-llm\phase01\corpus_train.txt")
c_test = dump(test_files, r"C:\Users\Geroin\chaotic-llm\phase01\corpus_test.txt")
print(f"files: {n}  train: {n_train} ({c_train/1e6:.2f} MB)  test: {n_test if False else len(test_files)} ({c_test/1e6:.2f} MB)")
print("train_files sample:", [os.path.basename(p) for p in train_files[:3]])
print("test_files sample:", [os.path.basename(p) for p in test_files[:3]])
