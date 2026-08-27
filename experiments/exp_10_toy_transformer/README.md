# Experiment 10 — Toy Chaotic Transformer (Tasks A–D)

## Hypothesis

Может ли архитектура «хаос-перемешивание + readout» решать toy-задачи сравнимые с attention? Какие компоненты дают эффект?

## Models

- **Chaotic+GSF**: 6 блоков (перестановка Арнольда + симметричное сцепление), GSF-гейты
- **Chaotic**: то же без GSF
- **ChaoticAttnReadout**: хаос + контент-зависимый readout на запросе (O(N))
- **GRU, LocalAttn (окно 5), FullAttn (2 слоя), MLP** — baseline

Все модели: d=32, one-hot вход, кросс-энтропия, 300–500 эпох.

## Results

### Task A — associative recall (N=16, V=16, chance 6.3%)

| Модель | Accuracy | FLOPs |
|---|---|---|
| FullAttn | **1.000** | 8.5e5 |
| LocalAttn | 0.415 | 6.9e5 |
| ChaoticAttnReadout | **0.915** | 3.7e4 |
| Chaotic+GSF | 0.194 | 3.2e4 |
| Chaotic | 0.200 | 3.8e3 |
| GRU | 0.163 | 2.0e5 |
| MLP | 0.066 | 1.3e5 |

### Task B — long-range, L=16 (N=25, см. exp_05)

Все модели 1.0 — задача была тривиальной (B=A, копирование). Честная версия — exp_05.

### Task C — hierarchical retrieval (N=16, V=32, chance 3.1%)

| Модель | Accuracy |
|---|---|
| Chaotic+GSF | **0.233** |
| LocalAttn | 0.144 |
| FullAttn | 0.143 |
| Chaotic | 0.101 |
| GRU / MLP | 0.029 |

### Task D — compositional/chain retrieval (N=9, V=16, chance 6.3%)

| Модель | Accuracy |
|---|---|
| Chaotic+GSF | **1.000** |
| LocalAttn | 1.000 |
| FullAttn | 1.000 |
| Chaotic | 0.139 |
| GRU / MLP | 0.062 |

## Interpretation

1. **Селективность — решающий фактор.** Blind-перемешивание + MLP-readout (Chaotic) ≈ 0.2 на задаче A: информация распространена, но извлечь нужную нечем. Контент-зависимый readout (ChaoticAttnReadout) даёт 0.915 при в ~9× меньших FLOPs, чем FullAttn (1.0).
2. **GSF-гейты помогают только там, где задача — «найти по маркеру»** (задача D: 1.0 vs 0.139 без GSF), но не дают селективности по содержимому (задача A: 0.19–0.2).
3. GRU проваливается на retrieve-задачах (позиционный readout не совпадает с его sequential-природой), но решает long-range (exp_05).
4. MLP (нет кросс-токен взаимодействий) — шанс везде, где нужна коммуникация: контрольный negative.

## Conclusion

**Частично подтверждено**: хаотическое перемешивание может заменить часть pairwise-взаимодействий attention (распространение), но селективность требует отдельного контент-зависимого механизма — и он может быть дешёвым (O(N) на запрос вместо O(N²)).

## Next

exp_11 — scaling, exp_12 — сравнение с attention, exp_13 — ablation.