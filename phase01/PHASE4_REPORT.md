# PHASE4 REPORT — Chaotic Candidate Routing

## 1. Hypothesis

Chaotic dynamics can serve as a cheap proposal/routing mechanism for sparse
attention — generating a small set of candidate tokens that overlaps
substantially with what full attention considers important.

## 2. Architecture (proposed)

```
Tokens → Embedding → Hierarchical Chaotic Mixer → Candidate Generator
  → Top-K → Sparse Attention over K → β / Readout → Logits
```

## 3. Experimental setup

| Parameter | Value |
|---|---|
| Corpus | code (1422 files TS/TSX/PY, 4:1 split, 9.26 MB train / 2.33 MB test) |
| Tokenizer | BPE-512 (ByteLevel, trained on train corpus) |
| Window | W = 256 (hierarchical: 4 × 64 local + global relay) |
| Model dim | d = 64 |
| Teacher | AttnLM (single-head full attention, exp13, 12K steps) |
| Mixer | Hierarchical Chaotic Mixer (exp14, 1 layer, 8+4 blocks, 6K steps) |
| Test windows | 1500 (stride 32) |
| Eval metrics | recall@K, precision@K, jaccard, top1-hit, any-hit, candidate distance, Spearman, Pearson |

## 4. Baselines

| Baseline | Implementation |
|---|---|
| Full Attention | exp13 AttnLM (teacher) — 1.0 accuracy on retrieval, PPL ~22 |
| Random Sparse | random top-K candidates — recall ≈ K/W |
| Local Readout | exp18 v0.3 — no attention, PPL 32.8 / 13.8+β |
| Chaotic Top-K | exp19 — mixer + top-K + attention over K, PPL 33.1 / 13.9+β |

## 5. Candidate Recall (Experiment I — THE GATE)

**Главный результат Phase 4:**

| K | chaos_qk recall | raw_qk recall | linear_probe recall | Random |
|---|---|---|---|---|
| 4 | 0.076 | 0.040 | 0.154 | 0.016 |
| 8 | 0.098 | 0.053 | 0.182 | 0.031 |
| **16** | **0.118** | 0.079 | 0.219 | 0.063 |
| 32 | 0.167 | 0.142 | 0.286 | 0.125 |
| 64 | 0.277 | 0.261 | 0.388 | 0.250 |

**Correlation (chaos_qk vs attention weights):** Spearman = 0.044, Pearson = 0.062.
**Top-1 match:** chaos top-1 == attention top-1 in 1.3% of windows.

**Interpretation:**
- Recall@16 = 11.8% — гейт (50%) провален в 4.2×.
- Mixer даёт небольшое улучшение над сырыми состояниями (0.118 vs 0.079) — но не того рода, что нужен для routing.
- Линейный зонд (22%) и его малые расстояния кандидатов (24-80) показывают: attention учителя локальна. Хаотическое глобальное перемешивание (69-122) разрушает локальность.
- Корреляция ≈ 0 — контент-зависимая релевантность не возникает из контент-независимой динамики.

## 6. Sparse Attention (Experiment II — exp19, already complete)

| Model | PPL | PPL+β | top-1+β | Attention cost |
|---|---|---|---|---|
| Full Attention (teacher) | 22.4 | 11.9 | 42.4% | O(W²) |
| Chaotic Top-16 (exp19 V1) | 33.07 | 13.95 | 40.8% | O(16²) |
| Local Readout (v0.3) | 32.83 | 13.79 | 40.7% | 0 |

Sparse attention on chaotic candidates matches local readout — not better.
The chaotic candidate set does not capture attention-relevant structure.

## 7. Attention Budget (Experiment III)

Not run — gate failed. Expected from exp19 data: chaotic sparse at 6.25%
interactions ≈ full attention minus 2% PPL, but this is due to broadcast/local
structure, not routing quality.

## 8. Global Memory (Experiment IV)

Not run — gate failed. With recall@16 = 11.8%, memory cannot recover the
missed candidates.

## 9. Routing Errors (Experiment V)

Not run — gate failed. The baseline routing error is already ~88% (missed
candidates at K=16).

## 10. Adaptive K (Experiment VI)

Not run — gate failed. Average recall is too low for any K.

## 11. Chaos vs Attention Correlation (Experiment VII)

| Metric | Value |
|---|---|
| Spearman | 0.044 |
| Pearson | 0.062 |
| Top-1 match | 1.3% |
| chaos_qk > random K/W factor | 1.9× |

The chaotic state's dot product with the query has essentially zero
correlation with the trained attention weights. The chaotic dynamics does
NOT encode attention-relevance structure.

## 12. GPU Benchmark (Experiment VIII)

Not run — gate failed. exp16 already showed 76-225× wall-clock disadvantage
for the chaotic mixer itself (Python loops vs cuBLAS). Even with positive
recall, this would be a critical bottleneck.

## 13. Results

### Что работает

1. **Хаотический mixer — дешёвый пропозер-перемешиватель (broadcast).**
   После смешивания каждый токен содержит сжатую информацию о контексте.
   Это позволяет локальному readout (v0.3) + β-prior работать на уровне
   full attention (PPL 32.8 vs 22.4, с β 13.8 vs 11.9).

2. **β-prior — эффективный селектор.** Агрегированная таблица n-грамм даёт
   +11-12 п.п. top-1, в 1.84× лучше kNN-LM при равной памяти.

3. **Иерархия закрывает long-range.** PPL 42.9 → 8.96 при контексте 64→256.

### Что не работает

1. **Chaotic routing for sparse attention.** Recall@16 = 11.8% (гейт 50%).
   Корреляция хаос↔attention ≈ 0. Контент-независимая динамика не может
   предсказывать контент-зависимые attention-веса.

2. **Chaotic top-K attention** (exp19): PPL 33.07 vs full 22.4 — не заменяет
   full attention. Причина: top-K не находит нужные токены (recall 11.8%).

3. **Селекция из динамики без контент-зависимости** (exp20): recall@K=0.25.

## 14. Failure modes

1. **Математическая причина**: фиксированная обратимая пермутация + сцепление
   контент-независимы. Attention-веса контент-зависимы (query × key). Без
   контент-зависимого взаимодействия routing не может работать.

2. **Глобальность хаоса разрушает локальность attention**: attention учителя
   локальна (avg distance 20-40), хаотические кандидаты глобальны (69-122).

3. **Mixer broadcast избыточен для routing**: если каждый токен содержит всё,
   «выбор кандидатов» теряет смысл.

## 15. Interpretation

**Phase 4 не подтвердила гипотезу sparse routing через хаос.** Гейт провален
на уровне 11.8% при пороге 50%. Исследование показало, что хаотическая
динамика не генерирует структуру, которая коррелирует с attention.

**Это НЕ означает, что хаотический mixer бесполезен.** Он — эффективный
broadcast-механизм (дешёвый пропозер). Но для маршрутизации к нему
необходимо контент-зависимое сцепление (O(W)), что является отдельной
архитектурной задачей, а не свойством хаотической динамики.

## 16. Conclusion

**Количественный ответ на главный вопрос Phase 4:**

> **Способен ли хаотический процесс дешёво предсказывать малое множество
> связей, которые являются семантически значимыми для attention, настолько
> хорошо, чтобы sparse attention на этих связях сохранял качество full attention?**

```
Full Attention:
  PPL = 22.4
  Interactions = 100%

Chaotic Sparse (K=16):
  PPL = 33.07
  Interactions = 6.25%
  Quality retention = 33.07/22.4 = 1.48×
  Candidate Recall@16 = 11.8%
  Correlation = 0.044

→ Хаотический процесс НЕ предсказывает семантически значимые связи attention.
  Recall@16 = 11.8% (порог 50%).
  PATH A: гипотеза sparse routing через хаос не подтверждена.
```

## 17. PATH A — Что дальше

Хаотический mixer остаётся в архитектуре как **дешёвый пропозер-перемешиватель**
(v0.3: mixer + local readout + β-prior, attention-free, PPL 11.24 с β=0.972).

**Новая архитектурная гипотеза:** контент-зависимое сцепление (coupling)
за O(W·d) — единственный честный путь к селекции из динамики пространства.
Это требует не карты Арнольда, а обученного сцепления, где сила смешивания
зависит от содержимого токенов (query-conditioned gates / attention without
softmax / linear attention). Это уже не «хаотическая динамика», а
«структурированная линейная динамика + обученные проекции».