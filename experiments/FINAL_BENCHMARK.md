# FINAL BENCHMARK — честная верификация STS-Prog

## Цель

Установить, что реально показывает текущая архитектура STS-Prog при честном сравнении с Transformer baseline.

Запрещено подгонять код, гиперпараметры, seed, выбор checkpoint или метрику под заранее ожидаемый результат.

Если STS-Prog проигрывает — это валидный результат и его нужно честно зафиксировать.

## Dataset

- **Файл:** `phase01/corpus_train.txt` (малый корпус, 487K токенов)
- **Tokenizer:** BPE-512, ByteLevel, без `add_prefix_space`
- **Vocab size:** 512
- **Sequence length (W):** 256
- **Train/validation split:** первые 80% — train, последние 20% — validation (фиксированное)
- **Training tokens:** ~390K (train split)

## Архитектуры

### STS-Prog

- **Класс:** `PurePCLM` из `models_pc.py`, `driver_mode="sts_prog"`
- **Hidden dim (d):** 192
- **Layers:** 8
- **Параметры:** ~900K (фиксируется в запуске)
- **Функция потерь:** CE + aux loss (multibead, w=0.5)

### STS-Prog no-PC

- **Класс:** `PurePCLM` из `models_pc.py`, `driver_mode="sts_prog_nopc"`
- **Hidden dim (d):** 192
- **Layers:** 8
- **Параметры:** ~900K

### Transformer

- **Класс:** `TransformerLM` из `parametric_models.py`
- **Hidden dim (D):** подбирается под ~900K при 8 слоях, 4 heads
- **Layers:** 8
- **Heads:** 4
- **Параметры:** ~900K (фиксируется)

## Training protocol

| Параметр | Значение |
|---|---|
| Optimizer | AdamW (weight_decay=0.01) |
| Learning rate | 5e-4 |
| LR scheduler | Linear warmup (1000 steps) + constant |
| Batch size | 64 |
| Gradient clipping | 1.0 |
| Steps | 6000 |
| Precision | fp16 (AMP) |
| Seed | 0, 1, 2 (3 seeds) |
| Checkpoint selection | best by validation PPL (последний, если не указано) |
| Data order | shuffled each epoch, fixed seed per run |

## Evaluation protocol

### PPL

- Validation loss over full validation split
- Mixer PPL = exp(mean(CE))
- Gated PPL = exp(mean(logaddexp(CE_logits, log_prior)))

### Retrieval

- **Задача:** induction (A→B...A→B)
- **Дистанции:** 16, 32, 64, 128, 256
- **Минимум:** 200 валидных trials на дистанцию
- **Формула:** accuracy = successful / total
- **Доверительный интервал:** Wilson score interval (95%)
- **Запрещено:** исключать неудачные trials, менять threshold, выбирать seed

### Scaling

Прогнать W ∈ {64, 128, 256, 512, 1024} (максимум на GPU):
- inference time (1 batch)
- peak VRAM
- tokens/sec

## Запуск

```bash
cd phase01/exp_vq
python final_benchmark.py
```

Результаты: `results/final_benchmark.json` + `results/final_benchmark.md`