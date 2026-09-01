# ChaoticLLM → STS-Prog: история поиска альтернативы attention

**Репозиторий:** экспериментальный поиск быстрой и экономичной архитектуры LLM,
альтернативной классическому transformer с attention.

## Краткая история (от чего к чему пришли)

### Phase 0–3: Хаос как вычислитель (Arnold cat map)
**Вопрос:** Может ли обратимая хаотическая динамика (карта кота Арнольда)
заменить часть pairwise attention?

**Ответ:** НЕТ (70+ экспериментов, все провалились).
- Arnold-карта — точная обратимая перестановка позиций, **но не перемешивает значения**
- Для коммуникации токенов нужен coupling, но он **уничтожает адресность/идентичность**
- Попытки «протащить идентичность через хаос» (PRIOR-PROP, BLACK-HOLE, КТО-селекторы,
  VQ, Link Keeper) — все провалились
- **Единственное работающее** — внешний exact prior (order-3, gated PPL ~9.5)

**Файлы:** `phase01/exp_01`–`exp_56`, `FINAL_REPORT.md`, `FINAL_REPORT_v2.md`
**Статус:** ❌ ЗАКРЫТО — Arnold не работает как замена attention

### Phase 4: Пекоры–Кэрролл (PC-синхронизация)
**Инсайт:** Для переноса идентичности через хаос нужна **диссипативная** динамика
(консервативная Arnold-карта не может синхронизироваться — det=1, нет устойчивого
многообразия).

**Результаты:**
- PC-синхронизация: sync_err=0, decode 100% на L=64 (впервые!)
- Чистый PC-микшер: mixer PPL 35.41 (vs Arnold 40.43, −12%)
- **Но:** retrieval всё ещё слабый (1-3%), селекция не работает

**Файлы:** `phase01/exp_vq/pc_probe*.py`, `PC_PROBE_REPORT.md`
**Статус:** ⚠️ Транспорт работает, селекция — нет

### Phase 5: STS-Prog (Sparse Temporal Selection with Progressive refinement)
**Прорыв:** Замена хаотической селекции на **content-addressable retrieval по сырым
эмбеддингам** + PC-синхронизация как усилитель retrieval.

**Ключевые открытия:**
- `sts_emb` (select на сырых эмбеддингах): mixer 25.23, retrieval 24-38%
- `sts_prog` (прогрессивное уточнение): mixer 19.22, **retrieval 47/40/35/35**
- `sts_prog_nopc` (без хаоса): mixer 23.99 — уже бьёт трансформер

**Head-to-head vs Transformer (одинаковый протокол, 8 слоёв):**

| Метрика | STS-Prog | Transformer | Разница |
|---|---|---|---|
| Mixer PPL | **19.22** | 24.40 | **−21%** |
| Retrieval L=16 | **47%** | 18% | **×2.6** |
| Retrieval L=256 | **35%** | 19% | **+84%** |
| Сложность | O(W·d·L) | O(W²·L) | дешевле |

**Файлы:** `phase01/exp_vq/models_pc.py`, `experiment_pc.py`, `match_transformer.py`
**Подробнее:** `phase01/exp_vq/README.md`
**Статус:** ✅ ПРОРЫВ — дешёвая альтернатива attention, работающая быстрее и точнее

---

## Структура репозитория

```
chaotic-llm/
├── README.md                              ← этот файл
├── .gitignore
│
├── phase01/                               ← все эксперименты
│   ├── exp_01…exp_56/                     ← Phase 0–3: Arnold, хаос, старые гипотезы
│   │   ├── experiment.py
│   │   └── results.json
│   ├── exp_vq/                            ← Phase 4–5: PC + STS-Prog (ПРОРЫВ)
│   │   ├── README.md                      ← подробное описание победителя
│   │   ├── models_pc.py                   ← архитектура PurePCLM + STS-Prog
│   │   ├── experiment_pc.py               ← протокол обучения + aux loss
│   │   ├── match_transformer.py           ← честный матч vs transformer
│   │   ├── pc_probe*.py                   ← пробы PC-синхронизации
│   │   ├── diagnose_las*.py               ← диагностика селекции
│   │   ├── link_keeper_probe.py           ← проба Link Keeper (FAIL)
│   │   ├── night_sts_prog.py              ← ночной прогон на Stack
│   │   ├── night_transformer.py           ← ночной матч трансформера
│   │   ├── chat_sts_prog.py               ← инференс/генерация
│   │   ├── results_*.json                 ← все результаты
│   │   └── night_ckpt_*.pt                ← чекпоинты
│   │
│   ├── corpus_train.txt                   ← малый корпус (487K токенов)
│   ├── corpus_stack_train.txt             ← Stack (2GB, 973M токенов) — только для ночных прогонов
│   ├── parametric_models.py               ← TransformerLM, вспомогательные модели
│   ├── chaos_lib.py                       ← библиотека хаотических карт
│   └── FINAL_REPORT.md                    ← отчёт Phase 0–3 (исторический)
│
├── experiments/                           ← Phase 0–3 toy-эксперименты (исторические)
│
├── FINAL_REPORT.md                        ← исторический
├── interactive_map.html                   ← визуализация (историческая)
└── [старые файлы: chaos_lib.py, phase2.py, …] — исторические, см. Phase 0–3
```

---

## Как воспроизвести победу

```bash
cd phase01/exp_vq

# 1. Базовая модель sts_prog (900K, малый корпус)
python experiment_pc.py pc --driver sts_prog --k 1.2 --sync-steps 8 --layers 8 --d 192 --aux-w 0.5 --aux-mode multibead

# 2. Матч vs трансформер (8 слоёв, выровненный протокол)
python match_transformer.py

# 3. Ночной прогон на Stack (100M токенов, 3.5M модель)
python night_sts_prog.py --d 384 --layers 12 --steps 20000 --sub 100000000 --batch 32

# 4. Инференс (генерация текста)
python chat_sts_prog.py --prompt "def fibonacci(n):" --steps 80 --temp 0.8
```

---

## Что дальше

### Ближайшие шаги
- [ ] Обучение на чат-датасете (вопрос→ответ) — чтобы модель «разговаривала»
- [ ] Интеграция с Hermes (локальный OpenAI-совместимый сервер)
- [ ] Чистка репозитория: архив старых экспериментов, единый README

### Долгосрочные направления
- [ ] Масштабирование (100M+ параметров, 1B+ токенов)
- [ ] Сравнение с transformer на длинных контекстах (W≥1024)
- [ ] Оптимизация через Triton / torch.compile

---

## Ключевые файлы победителя

| Файл | Что это |
|---|---|
| `phase01/exp_vq/models_pc.py` | Архитектура PurePCLM + все режимы селекции |
| `phase01/exp_vq/experiment_pc.py` | Протокол обучения, aux loss, eval |
| `phase01/exp_vq/match_transformer.py` | Честный матч vs трансформер |
| `phase01/exp_vq/README.md` | Подробное описание победителя |
| `phase01/exp_vq/chat_sts_prog.py` | Инференс/генерация |