"""Smoke-тесты: проверяют измерительный код, а не качество модели.

Запуск:  python -m pytest tests/ -q
Без GPU и без корпуса — всё на CPU и на синтетике.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VQ = os.path.join(ROOT, "phase01", "exp_vq")
sys.path.insert(0, ROOT)
sys.path.insert(0, VQ)
sys.path.insert(0, os.path.join(ROOT, "phase01"))
sys.path.insert(0, os.path.join(ROOT, "phase01", "exp_memory_selector"))

from models_pc import build_pc_model, W  # noqa: E402
import benchmark  # noqa: E402


def tiny_model(driver="sts_prog", vocab=64, d=32, layers=2):
    torch.manual_seed(0)
    return build_pc_model("pc", vocab, d=d, alpha=0.3, k_init=1.2, sync_steps=1,
                          driver_mode=driver, temp=0.3, layers=layers)


def test_forward_shape():
    for driver in ("sts_prog", "sts_prog_nopc"):
        m = tiny_model(driver)
        x = torch.randint(0, 64, (2, W))
        out = m(x)
        assert out.shape == (2, 64), f"{driver}: форма выхода {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{driver}: в логитах NaN/inf"


def test_nopc_really_removes_chaos():
    """Абляция обязана отличаться от полной модели, иначе она ничего не доказывает."""
    torch.manual_seed(0)
    a = tiny_model("sts_prog")
    torch.manual_seed(0)
    b = tiny_model("sts_prog_nopc")
    x = torch.randint(0, 64, (2, W))
    with torch.no_grad():
        da = a(x) - b(x)
    assert da.abs().max().item() > 0, "nopc выдаёт тот же результат, абляция фиктивная"


def synthetic_ids(n=6000, vocab=64, seed=0):
    """Корпус с гарантированными повторами, чтобы retrieval-тест находил пары."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=n).astype(np.int64)
    for i in range(0, n - 300, 37):
        ids[i] = 7
        ids[i + 1] = 9
    return ids


class ConstantModel(torch.nn.Module):
    """Заглушка: всегда предсказывает один токен. Нужен, чтобы тестировать метрику."""

    def __init__(self, vocab, answer):
        super().__init__()
        self.vocab = vocab
        self.answer = answer

    def forward(self, x):
        b = x.shape[0]
        out = torch.full((b, self.vocab), -10.0)
        out[:, self.answer] = 10.0
        return out

    def eval(self):
        return self


def test_retrieval_records_denominator():
    """Знаменатель обязан попадать в результат — иначе 67% может быть «2 из 3»."""
    ids = synthetic_ids()
    m = ConstantModel(vocab=64, answer=9)
    res = benchmark.induction_retrieval(m, ids, distances=(16, 64),
                                        n_valid_target=30, max_attempts=200,
                                        device="cpu")
    for L, r in res.items():
        assert "n_valid" in r and "hits" in r and "rate" in r, f"{L}: нет полей"
        assert r["n_valid"] > 0, f"{L}: знаменатель нулевой, метрика не интерпретируема"
        assert r["hits"] <= r["n_valid"], f"{L}: hits больше знаменателя"
        assert abs(r["rate"] - r["hits"] / r["n_valid"]) < 1e-9, f"{L}: доля не бьётся"


def test_retrieval_no_silent_division_when_no_valid_trials():
    """Если подходящих проб нет, должен быть None, а не ноль и не деление на 1."""
    ids = np.zeros(4000, dtype=np.int64)  # совпадения есть, но пар A→B на расстоянии нет
    m = ConstantModel(vocab=64, answer=3)
    res = benchmark.induction_retrieval(m, ids, distances=(256,),
                                        n_valid_target=5, max_attempts=3,
                                        device="cpu")
    r = res["L256"]
    assert r["n_valid"] == 0 and r["rate"] is None, f"ожидали None, получили {r}"


def test_benchmark_matrix_has_unique_names():
    names = [c["name"] for c in benchmark.MATRIX]
    assert len(names) == len(set(names)), "имена конфигов в матрице повторяются"


def test_benchmark_matrix_references_existing_drivers():
    from experiment_pc import build_pc_model as _b  # noqa: F401
    allowed = {"mean", "last", "top1", "soft", "crt", "sts_emb", "sts_h",
               "sts_lq", "sts_lqk", "sts_prog", "sts_prog_nopc", "__transformer__"}
    for c in benchmark.MATRIX:
        assert c["driver"] in allowed, f"{c['name']}: неизвестный driver {c['driver']}"


@pytest.mark.skipif(not os.path.isdir(os.path.join(VQ, "benchmark")),
                    reason="папка benchmark/ появится после прогона")
def test_benchmark_jsons_have_full_provenance():
    d = os.path.join(VQ, "benchmark")
    # Проверяем только канонические артефакты по сидам; _agg / scaling пишутся в
    # другой схеме и здесь не проверяются.
    files = [f for f in os.listdir(d) if f.endswith(".json") and "_seed" in f]
    if not files:
        pytest.skip("прогон ещё не дал результатов")
    for fn in files:
        with open(os.path.join(d, fn)) as f:
            rec = json.load(f)
        for key in benchmark.REQUIRED_KEYS:
            assert key in rec, f"{fn}: нет поля {key}"
        for L, r in rec["retrieval"].items():
            assert r.get("n_valid", 0) > 0, f"{fn}/{L}: знаменатель не записан"
        # corpus.n_train_tokens может быть None, если прогон его не логировал —
        # важно, что сам блок corpus присутствует (пустота корпуса ловится выше).
        assert "corpus" in rec["provenance"], f"{fn}: нет блока provenance.corpus"
