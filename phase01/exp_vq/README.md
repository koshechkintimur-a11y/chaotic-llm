# STS-Prog: дешёвый content-addressable retrieval-контур + residual-микшер

**Ветка:** `pc-carroll` | **Дата:** 2026-08-31 | **Честная формулировка результата**

## Что это

**Не «хаос как вычислитель».** Это **гибрид**:
- **Content-addressable retrieval-контур** на сырых эмбеддингах (select-then-sync, top-k по cosine) — главный вклад в retrieval.
- **Residual-микшер** (диссипативная хаотическая динамика PC) — уточнение query и bulk-mixing смысла.
- **Multi-input readout** (h_last + q0 + global mean).

**Сложность:** `O(W·d·L)` против attention `O(W²·L)`, и **без FFN** (50% стоимости слоя трансформера).

## Ключевые цифры (одинаковый бюджет ~900K параметров, тот же корпус)

| Метрика | **sts_prog** | Transformer (D=124, L=4, H=4) | Итог |
|---|---|---|---|
| mixer PPL | **23.95** | 25.53 | **−6.2%** |
| gated PPL (order-3) | **8.62** | 9.39 | **−8.2%** |
| retrieval L=16 | **0.396** | 0.201 | **×2** |
| retrieval L=256 | **0.317** | 0.191 | **+66%** |
| время 6000 шагов | **~297с** | 478с | **−38%** |

**⚠️ Протокол-мачта не идеален:** 8 блоков против 4 слоёв, LR 5e-4+warmup против 3e-4 (трансформер не тюнен). Победа реальна, но **не «паритет»**. Тюнен трансформера — TODO.

## Атрибуция (ЧЕСТНО, по ревью)

**Что даёт хаос (PC-блоки), а что — контур селекции?** См. абляцию `sts_prog_nopc`
(блоки → identity, контур + readout остаются). **Без неё победу атрибутировать нельзя.**
Косвенно: mean 36.6 → sts_emb 30.35 → sts_prog 23.95 — основной вклад даёт контур.

**Аудит (необученная модель):** контур даёт retrieval 13.2% на L=16 — **17-34× выше шанса
до единого шага обучения**. Причина: два вхождения A имеют одинаковый эмбеддинг,
cosine ≈ 0.5 против ≈ 0. Примерно половина retrieval — **подарок архитектуры**.

**Мёртвые параметры (найденные ревью):**
- `alpha` принимается, нигде не используется.
- `self.k` в sts_prog мёртв — работает захардкоженный k=1.2 из PurePCBlock.
- При k=1.2: `h_new = 1.2·driver − 0.2·h` — это перелёт, не стягивание. TODO: чинить.

**FLOPs-честность:** O(W²)-член (QKᵀ+@V) — всего 26% стоимости слоя трансформера,
FFN — 50%. Выигрыш в основном от **отсутствия FFN**. У хаоса `h @ W` — 97.6% его стоимости
(O(W·d²) у обеих моделей).

## История

1. Arnold-микшер (baseline) — PPL 40.4, retrieval 0–5%.
2. **Can reversible chaos (Arnold) replace attention? НЕТ** — 70+ экспериментов провал.
   Карта консервативна (det=1, собственные 0.38/2.62, пересечение диапазонов k пусто).
3. Чистый PC без Arnold (диссипативный) — PPL 35.4, retrieval 1–3%.
4. sts_emb — PPL 25–30, retrieval 24–38%.
5. **Can dissipative chaos + sparse attention beat transformer? ДА** — sts_prog побеждает.

## Воспроизведение

```bash
cd phase01/exp_vq
python experiment_pc.py pc --driver sts_prog --k 1.8 --sync-steps 8 --layers 8 --d 192 --aux-w 0.5 --aux-mode multibead
python experiment_pc.py pc --driver sts_prog_nopc --k 1.8 --sync-steps 8 --layers 8 --d 192 --aux-w 0.5 --aux-mode multibead  # абляция
python match_transformer.py
```

## Файлы

| Файл | Назначение |
|---|---|
| `models_pc.py` | PurePCLM: mean/top1/sts_emb/sts_prog/sts_prog_nopc |
| `experiment_pc.py` | тренировка + eval + aux loss |
| `match_transformer.py` | матч vs TransformerLM |
| `diagnose_las2.py` | диагностика точности селекции (Q1/Q2) |
| `_audit_diag.py` | аудит необученной модели (вклад архитектуры vs обучения) |
