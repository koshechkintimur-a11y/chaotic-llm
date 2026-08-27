# Experiment 12 — Chaotic Routing vs Attention

## Hypothesis

Специальный эксперимент §21: N токенов + один QUERY + один TARGET. Attention взаимодействует со всеми; хаотическая модель получает X₀ + control + T итераций. Сравнить Cost при равной Accuracy.

## Setup

Задача A (associative recall, N=16, V=16). Модели обучены 500 эпох:
- ChaoticAttnReadout: 6 хаос-блоков + attention readout на запросе (O(N))
- Chaotic: 6 хаос-блоков + MLP readout
- FullAttn: 2 слоя полного self-attention (все пары)

## Results

| Модель | Accuracy | FLOPs (на sample) | FLOPs/accuracy |
|---|---|---|---|
| **ChaoticAttnReadout** | **0.915** | **3.66e4** | 4.0e4 |
| Chaotic (MLP readout) | 0.200 | 3.84e3 | 1.9e4 |
| FullAttn | 1.000 | 3.40e5 | 3.4e5 |

## Interpretation

- **Область «та же точность при меньшем Compute» найдена**: 0.915 vs 1.000 (Δ=0.085) при 9.3× меньших FLOPs.
- ВАЖНАЯ ОГОВОРКА: ChaoticAttnReadout выигрывает за счёт того, что readout контент-зависимый, но **дешёвый** — O(N·d) на запрос (один query), а не O(N²) на все пары. Это не «хаос дешевле attention», а «структурированное перемешивание + селективный readout дешевле all-pairs».
- Для задач, где отвечать должен КАЖДЫЙ токен (языковое моделирование), потребуется N query-проходов → O(N²), преимущество исчезает.

## Conclusion

**Частично подтверждено**: для retrieval-задач с одним запросом гибрид «хаотическое перемешивание + селективный readout» даёт близкую к attention точность при ~10× меньших FLOPs. Механизм экономии — не хаос, а замена all-pairs на «распространить + выбрать на запросе».

## Next

exp_13 — ablation.