# Experiment 16 — Honest GPU Latency Benchmark (Step 5c)

## Hypothesis

Асимптотическое FLOPs-преимущество O(W log W) vs O(W²) конвертируется
в wall-clock-преимущество на GPU на всех W.

## Setup

- GPU: RTX 3060 (12GB). Torch 2.5.1+cu121, CUDA 12.1.
- Models: Hierarchical Chaotic Mixer (Python-loop, 8 local + 4 global blocks)
  vs Full Attention (1 head, single fused matmul).
- W = 512, 1024, 2048. Batch 1 and 8.
- Warmup 10, measure 50 forward passes, sync CUDA, report mean ms.

## Results

| W | FLOPs ratio (chaos/attn) | B=1 chaos ms | B=1 attn ms | Latency ratio |
|---|---|---|---|---|
| 512 | 864K vs 55M (1:63) | 58.08 | 0.76 | **76× SLOWER** |
| 1024 | 1.7M vs 210M (1:122) | 90.10 | 0.73 | **123× SLOWER** |
| 2048 | 3.4M vs 822M (1:239) | 178.00 | 0.79 | **225× SLOWER** |

## Interpretation (CRITICAL)

1. **FLOPs-преимущество реально и растёт с W** (63× → 239×).
2. **Но реализация на GPU уничтожает выигрыш**: attention — один fused matmul
   (cuBLAS), а хаотический миксер — Python-циклы, gather-индексы, невекторизованные
   операции. На GPU внимание в 76-225× быстрее на wall-clock.
3. **Breakeven-точка** на кривой O(W log W) vs O(W²) с текущей имплементацией
   лежит за W > 100K (там attention займёт ~ 100× W=2048 = 79ms → равно хаосу).
4. **Вывод**: для практического выигрыша необходимы fused CUDA-ядра для
   хаотического перемешивания + сцепления, КАРДИНАЛЬНО снижающие constant factor.

## Conclusion

**Опровергнуто** для практических размеров W ≤ 2048 при текущей реализации.
Асимптотический выигрыш есть, но constant factor от неоптимальной реализации
на GPU его полностью съедает. Требуется fused CUDA kernel.