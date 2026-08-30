"""retrieval_honest.py — честный тест дальних связей (индукционная головка).

Проблема старого теста (retrieval_accuracy): KEY вставлялся СЛУЧАЙНЫМ токеном —
модель никогда не видела такого паттерна, даже морен давал 0. Это проверяло
«копирование случайного токена», а не память.

Честный тест — индукционная головка на РЕАЛЬНЫХ паттернах корпуса:
  [A, B, ..., A, ?]  где A встречается снова на дистанции L внутри окна.
Модель должна вспомнить «после A идёт B» и предсказать B.

Метрика: acc (доля, где argmax == B) по дистанциям L, и lift vs baseline
(acc на случайных позициях). lift > 1 → модель использует дальние связи.

Usage: python retrieval_honest.py H5
"""
import os
import sys
import json
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, REPO)

from experiment import W, V, DEVICE, test_ids
from models import build_model


def collect_induction(ids, distances=(16, 64, 128, 256), n_pat=1500):
    """Реальные индукционные паттерны: A->B, потом A снова на дистанции L,
    цель = B (что шло после первого A)."""
    rng = np.random.default_rng(11)
    pats = {L: [] for L in distances}
    tries = 0
    maxstart = len(ids) - 2
    while sum(len(v) for v in pats.values()) < n_pat and tries < 400000:
        tries += 1
        i = int(rng.integers(0, maxstart))
        A = ids[i]
        B = ids[i + 1]
        if A == B or B == 0:
            continue
        # ищем повторение A на дистанции L в {16,64,128,256} (вперёд)
        for L in distances:
            j = i + L
            if j + 1 >= len(ids):
                continue
            if ids[j] == A and len(pats[L]) < n_pat // len(distances):
                pats[L].append((j, B))     # позиция второго A, цель B
                break
    return pats


def run(config, n_pat=1500):
    model = build_model(config, vocab=V)
    ckpt = os.path.join(HERE, f"model_{config}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    pats = collect_induction(test_ids, n_pat=n_pat)
    total = sum(len(v) for v in pats.values())
    print(f"[{config}] induction patterns: {total}", flush=True)
    if total == 0:
        print("  no patterns found")
        return

    # baseline: acc на случайных позициях (без индукции)
    rng = np.random.default_rng(5)
    base_hits = 0
    n_base = 0
    with torch.no_grad():
        for _ in range(400):
            i = int(rng.integers(W, len(test_ids) - 1))
            ctx = test_ids[i - W:i]
            X = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
            logits = model(X).cpu().numpy()[0]
            if int(np.argmax(logits)) == test_ids[i]:
                base_hits += 1
            n_base += 1
    base_acc = base_hits / n_base
    print(f"  baseline acc={base_acc:.3f}", flush=True)

    res = {}
    for L in (16, 64, 128, 256):
        hits = 0
        n = 0
        with torch.no_grad():
            for j, B in pats[L]:
                if j < W:
                    continue
                ctx = test_ids[j - W:j]          # окно, A на дистанции L внутри
                X = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
                logits = model(X).cpu().numpy()[0]
                if int(np.argmax(logits)) == B:
                    hits += 1
                n += 1
        acc = hits / max(1, n)
        res[L] = {"acc": acc, "n": n, "lift": acc / max(1e-6, base_acc)}
        print(f"  L={L}: acc={acc:.3f} (n={n}) lift={res[L]['lift']:.2f}", flush=True)

    json.dump({"config": config, "baseline_acc": base_acc, "by_L": res},
              open(os.path.join(HERE, f"retrieval_honest_{config}.json"), "w"), indent=2)
    print(f"saved retrieval_honest_{config}.json", flush=True)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "H5"
    run(config)
