# STS-Prog: дешёвый content-addressable retrieval-контур + residual-микшер

**Ветка:** `pc-carroll` | **Дата:** 2026-08-31 | **Статус:** ✅ ревью закрыто, прорыв подтверждён

> **Название архитектуры — STS-Prog, НЕ ChaoticLLM.**
> ```
> Исходный вопрос: Can reversible chaotic dynamics (Arnold) replace attention?
> Ответ: НЕТ (70+ экспериментов)
>
> Новый вопрос: Can dissipative chaos + sparse attention beat transformer?
> Ответ: ДА (STS-Prog побеждает на всех метриках)
>
> Архитектура: STS-Prog (не ChaoticLLM)
> - Sparse attention на сырых эмбеддингах (content-addressable retrieval)
> - Dissipative chaotic dynamics (PC-синхронизация)
> - Progressive query refinement + multi-input readout
> ```

## Что это

**Не «хаос как вычислитель».** Это **гибрид**:
- **Content-addressable retrieval-контур** на сырых эмбеддингах (select-then-sync, top-k по cosine) — основа.
- **Residual-микшер** (диссипативная хаотическая динамика PC) — уточнение query и bulk-mixing смысла.
- **Multi-input readout** (h_last + q0 + global mean).

## Градация: от чего к чему пришли

| Этап | Что делали | Mixer PPL | Retrieval L16 | Вывод |
|---|---|---|---|---|
| **exp52** | Arnold-микшер (baseline) | 40.43 | ~0% | Хаос-перестановка не работает |
| **PC-mean** | Диссипативный PC вместо Arnold | 35.41 | 1-3% | Транспорт заработал, селекция нет |
| **sts_emb** | Select на сырых эмбеддингах | 25.23 | 24-38% | Селекция — ключ |
| **sts_prog** | Прогрессивное уточнение | 23.95 | 40% | Первый баланс retrieval |
| **sts_prog+fix k** | Стягивание вместо перелёта | **19.22** | **47%** | 🏆 ПРОРЫВ |
| **Transformer** | Матч (8 слоёв, 904K) | 24.40 | 18% | Мы −21% PPL, ×2.6 retrieval |

**Атрибуция (честная, из абляции):**
- Контур селекции один (sts_prog_nopc): PPL 23.99 — уже бьёт трансформер
- Хаос (PC-блоки) добавляет: −25% PPL и ×2.6 retrieval

## Результаты head-to-head vs Transformer

| Метрика | STS-Prog | Transformer | Разница |
|---|---|---|---|
| Mixer PPL | **19.22** | 24.40 | **−21%** |
| Retrieval L=16 | **47%** | 18% | **×2.6** |
| Retrieval L=64 | **40%** | 17% | **×2.4** |
| Retrieval L=128 | **35%** | 21% | **+67%** |
| Retrieval L=256 | **35%** | 17% | **×2.1** |
| Сложность | O(W·d·L) | O(W²·L) | дешевле |
| Параметры | 900K | 904K | паритет |

> **Ночной прогон (Stack, ~100M токенов) удалён из описания победителя.** Его артефакты
> (`results_night_*.json`, `night_ckpt_*.pt`) сохранены в `phase01/exp_vq/` как экспериментальные,
> **не валидированы** (несогласованная глубина 12 vs 8, ненадёжный знаменатель retrieval) и
> **вытеснены** строгим v2-прогоном на публичном корпусе (`final_benchmark_v2.py`). Числа
> 25.65 / 67% / 60% не заявляются как доказательство и не входят в основные таблицы.

**Сложность:** `O(W·d·L)` против attention `O(W²·L)`, и **без FFN** (50% стоимости слоя трансформера).

## Ключевые цифры (одинаковый бюджет ~900K, выровненный протокол: 8 слоёв, LR 5e-4+warmup+clip)

| Метрика | **sts_prog** | Transformer (D=88, L=8, H=4) | Итог |
|---|---|---|---|
| mixer PPL | **19.22** | 24.40 | **−21%** |
| gated PPL (order-3) | **8.27** | 9.24 | **−10%** |
| retrieval L=16 | **0.472** | 0.181 | **×2.6** |
| retrieval L=64 | **0.404** | 0.166 | **×2.4** |
| retrieval L=128 | **0.348** | 0.210 | **×1.7** |
| retrieval L=256 | **0.349** | 0.175 | **×2.0** |
| время 6000 шагов | ~5 мин | 469с | близко |

**⚠️ Матч выровнен:** 8 слоёв у обоих, LR 5e-4 + warmup 1000 + grad clip 1.0 + N_EVAL 5000.

## Атрибуция (по ревью, честно)

| Конфиг | mixer PPL | retrieval L16 | Вклад |
|---|---|---|---|
| **Контур селекции один** (nopc) | 23.99 | 18% | основа (уже бьёт tf 24.40) |
| **+ хаос (sts_prog)** | **19.22** | **47%** | −25% PPL, ×2.6 retrieval |

**Хаос — полноценный компонент, когда параметры правильные.** До фикса k=1.2 (перелёт) вклад был вдвое меньше. После фикса на стягивание (sigmoid → k=0.77) — скачок.

**Мёртвые параметры починены:** alpha (сила хаоса в блоке), k (обучаемый, sigmoid → стягивание).


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
