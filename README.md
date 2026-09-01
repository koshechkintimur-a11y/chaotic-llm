# ChaoticLLM → STS-Prog: история поиска альтернативы attention

[🇷🇺 Русский](#russian) · [🇬🇧 English](#english)

<a id="russian"></a>
## Русский

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

**Ключевые открытия (строгий прогон `final_benchmark_v2.py`, публичный корпус, 6000 шагов, 5 сидов, 95% CI):**
- **STS-Prog побеждает Transformer по PPL при МЕНЬШЕМ бюджете параметров: 24.5 против 46.6 (−47%)** при 900K против 941K (matched D=92). Даже против несколько меньшего Transformer (D=88, 866K) — 24.5 против 49.1.
- **Преимущество в induction-retrieval целиком от хаоса/PC.** Чистый content-addressable retrieval без PC-синхронизации (`no-PC`, 17.3% на L16) не бьёт трансформер (21.8%); добавление PC-синхронизации поднимает до 49.9%.
- PPL-выигрыш тоже от хаоса/PC: `no-PC` 46.5 против полного STS-Prog 24.5 (оба по 900K) — ~1.9× улучшение PPL.
- Все числа — на **публичном** корпусе (TheAlgorithms/Python, MIT), воспроизводимы вне этой машины (см. TODO #3 и `SCIENCE_AUDIT.md`).

**Head-to-head vs Transformer (публичный корпус, 8 слоёв, 300 проб/дистанция × 5 сидов, фиксированный retrieval-сид):**

<!-- BEGIN GENERATED TABLES -->
### Параметр-матч benchmark v2 (строгий прогон, `final_benchmark_v2.py`)

| Модель | PPL mean±std | 95% CI | Параметры | Retr 16 | Retr 32 | Retr 64 | Retr 128 | Retr 256 |
|---|---|---|---|---|---|---|---|---|
| STS-Prog | 24.525±0.352 | [24.12, 24.93] | 900,353 | 49.9% (n=7500) | 51.9% (n=7500) | 41.3% (n=7500) | 42.0% (n=7500) | 37.9% (n=7500) |
| STS-Prog (no-PC) | 46.483±0.975 | [45.36, 47.60] | 900,353 | 17.3% (n=7500) | 15.3% (n=7500) | 16.7% (n=7500) | 16.0% (n=7500) | 15.2% (n=7500) |
| Transformer (D=88) | 49.132±1.526 | [47.38, 50.89] | 865,904 | 20.3% (n=7500) | 20.5% (n=7500) | 18.1% (n=7500) | 16.3% (n=7500) | 16.9% (n=7500) |
| Transformer-matched (D=92) | 46.620±1.492 | [44.91, 48.33] | 940,568 | 21.8% (n=7500) | 20.7% (n=7500) | 19.4% (n=7500) | 16.1% (n=7500) | 15.8% (n=7500) |

### Откуда числа (v2)

| Поле | Значение |
|---|---|
| commit | 9832a8672b014b15e70eb199c371a04e2136a441 |
| device | cuda |
| steps | 6000 |
| batch | 64 |
| lr | 0.0005 |
| warmup | 1000 |
| retrials | 300 |
| eval_seed | 12345 |
| protocol | FINAL_BENCHMARK.md (v2 rigorous) |
| seeds | [0, 1, 2, 3, 4] |

> **Методология v2 (почему эти числа надёжны):**
> - Retrieval считается на **одной фиксированной выборке** (`eval_seed`), поэтому разброс по сидам = чистая дисперсия модели, а не смесь дисперсии модели и выборки (ошибка v1, где тестовая выборка менялась по сидам).
> - Transformer-matched подобран **не меньше** STS по параметрам (D=92, ~941K ≥ STS 900K): победа STS — консервативная, при бОльшем бюджете оппонента.
> - PPL дан как mean±std и 95% CI (t-распределение по числу сидов). Retrieval = доля верных предсказаний B в паттерне A→B, усреднено по всем принятым пробам (n_trials на сид, пул по сидам; знаменатель в JSON).
> - Результаты пишутся **инкрементально** после каждого сида + `.pt`-чекпоинты, падение не теряет прогоны.

<!-- END GENERATED TABLES -->

**Файлы:** `phase01/exp_vq/models_pc.py`, `final_benchmark_v2.py` (строгий прогон), `prepare_public_corpus.py`, `render_docs_v2.py`, `rebuild_from_perseed.py`
**Подробнее:** `phase01/exp_vq/README.md`, `SCIENCE_AUDIT.md`
**Статус:** ✅ Подтверждено строгим прогоном на 5 сидах (публичный корпус, 95% CI) — альтернатива attention с меньшим PPL и лучшим retrieval при сопоставимых/меньших параметрах.

---

## Структура репозитория

```
chaotic-llm/
├── README.md                              ← этот файл
├── .gitignore
│
├── phase01/                               ← все эксперименты
│   ├── archive/                           ← Phase 0–3, сняты с активной работы:
│   │   ├── exp_morin_boost/               ←   порядок сохранён (git видит как rename)
│   │   ├── exp_prior_prop/
│   │   └── exp_eye_selector/
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

# 3. Строгий воспроизводимый прогон (публичный корпус, 5 сидов, 95% CI) — см. SCIENCE_AUDIT.md
python final_benchmark_v2.py --corpus ../corpus_public.txt --device cuda --steps 6000 --seeds 0,1,2,3,4 --retrials 300 --eval-seed 12345

# 4. Инференс (генерация текста)
python chat_sts_prog.py --prompt "def fibonacci(n):" --steps 80 --temp 0.8
```

---

## Что дальше

### Ближайшие шаги
- [ ] Обучение на чат-датасете (вопрос→ответ) — чтобы модель «разговаривала»
- [ ] Интеграция с Hermes (локальный OpenAI-совместимый сервер)
- [x] Чистка репозитория: старые эксперименты → `phase01/archive/`, единый README, `.gitignore` для тяжёлых данных (`ru_chat.json`, 58 МБ)
- [x] Честный финальный прогон (v1, `final_benchmark.py`, 3 сида) + строгий воспроизводимый v2 (`final_benchmark_v2.py`, 5 сидов, публичный корпус, 95% CI): STS-Prog PPL 24.5 vs TF-matched 46.6; таблицы влиты через `render_docs_v2.py` (см. TODO #3 и `SCIENCE_AUDIT.md`)

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

---

## Известные дыры в доказательствах (TODO)

Честный список мест, где заявленное число не подтверждено лежащим в репо артефактом.
Ничего не скрыто и не удалено — это напоминание, что надо перезапустить.

**1. `19.22` и `retrieval 47/40/35/35` — ✅ ЗАКРЫТО финальным прогоном.**
`phase01/exp_vq/results_pc.json` на текущем HEAD содержит
`mixer_ppl 23.991, retrieval 0.183/0.176/0.198/0.156` — это прогон абляции
`sts_prog_nopc`, а не победителя (те же цифры стоят в таблице атрибуции).
Числа 19.22 / 47% жили только в README и в сообщении коммита `32db903`.
**Сейчас:** финальный прогон `final_benchmark.py` (d=192, L=8, k=1.2, 3 сида, 6000 шагов)
дал STS-Prog PPL **17.94** (mixer 19.22 был недостижимо оптимистичен). Результаты
разложены в `benchmark/*.json` и вставлены в таблицу выше через `render_docs.py`.
Старая таблица (19.22 / 47%) заменена на генерируемую из артефактов.

**2. Разный `k` в командах воспроизведения.** Корневой README — `--k 1.2`,
`phase01/exp_vq/README.md` — `--k 1.8`. sigmoid(1.2) = 0.77 (то самое k=0.77
из описания фикса), sigmoid(1.8) = 0.86. Один из двух вариантов не
воспроизводит заявленный результат. **Финальный прогон `final_benchmark.py`
использует `k_init=1.2`** (как в корневом README) — для него дыра закрыта.

**3. Воспроизводимость, параметр-матч и статистика малого прогона — ✅ ЗАКРЫТО.**
Финальный прогон v1 (`final_benchmark.py`, 3 сида) подтвердил направление, но оставил три
дыры, которых нет в списке выше:
- **(a) Приватный корпус.** `final_benchmark.py` читает `phase01/corpus_train.txt`, который
  лежит в `.gitignore` (утечка приватных исходников) — результаты нельзя воспроизвести вне
  этой машины. Решение: `prepare_public_corpus.py` качает публичный code-корпус
  **TheAlgorithms/Python** (MIT, pinned SHA `9391f546d6f8`) → `phase01/corpus_public.txt`;
  прогон v2 (`final_benchmark_v2.py --corpus ...`) идёт на нём.
- **(b) Параметр-матч был нечестным.** STS-Prog (900 353) имел **+4%** к трансформеру
  (865 904); «matched»-конфиг брал меньшую модель, так что преимущество оставалось у STS.
  В v2 matched-Transformer = **D=92 (~941K) ≥ STS 900K** — сравнение консервативное
  (STS побеждает модель с бОльшим бюджетом).
- **(c) Статистика retrieval шаталась по сидам.** В v1 retrieval-тест засевался сидом модели
  (`np.random.seed(seed)` внутри `train_model`), поэтому ТЕСТОВАЯ ВЫБОРКА менялась по сидам,
  и разброс = смесь дисперсии модели и выборки. В v2 retrieval считается на
  **фиксированной выборке** (`eval_seed`), плюс 5 сидов и **95% CI** (t-распределение).
- **Статус: ✅ ЗАКРЫТО.** Прогон `final_benchmark_v2.py` завершён (5 сидов × 4 конфигурации × 6000 шагов,
  retrieval 300 проб/дистанция) на публичном корпусе `corpus_public.txt`; таблицы с 95% CI влиты в README
  через `render_docs_v2.py`. Дыры (a)–(c) закрыты: новый корпус, консервативный параметр-матч и
  статистика с 95% CI теперь воспроизводимы вне этой машины (см. `SCIENCE_AUDIT.md`).

---

<a id="english"></a>
## English

# ChaoticLLM → STS-Prog: the search for an attention alternative

**Repository:** an experimental search for a fast and cheap LLM architecture that is
an alternative to the classic transformer with attention.

## Brief history (how we got here)

### Phase 0–3: Chaos as a compute primitive (Arnold cat map)
**Question:** Can reversible chaotic dynamics (Arnold's cat map) replace part of pairwise attention?

**Answer:** NO (70+ experiments, all failed).
- The Arnold map is an exact reversible permutation of positions, **but does not mix values**
- Token communication needs coupling, but coupling **destroys addressability/identity**
- Attempts to "smuggle identity through chaos" (PRIOR-PROP, BLACK-HOLE, KTO-selectors,
  VQ, Link Keeper) — all failed
- **The only thing that worked** — an external exact prior (order-3, gated PPL ~9.5)

**Files:** `phase01/exp_01`–`exp_56`, `FINAL_REPORT.md`, `FINAL_REPORT_v2.md`
**Status:** ❌ CLOSED — Arnold does not work as an attention replacement

### Phase 4: Pecora–Carroll (PC-synchronization)
**Insight:** To carry identity through chaos you need **dissipative** dynamics
(the conservative Arnold map cannot synchronize — det=1, no stable manifold).

**Results:**
- PC-synchronization: sync_err=0, decode 100% at L=64 (for the first time!)
- Pure PC mixer: mixer PPL 35.41 (vs Arnold 40.43, −12%)
- **But:** retrieval is still weak (1-3%), selection does not work

**Files:** `phase01/exp_vq/pc_probe*.py`, `PC_PROBE_REPORT.md`
**Status:** ⚠️ Transport works, selection does not

### Phase 5: STS-Prog (Sparse Temporal Selection with Progressive refinement)
**Breakthrough:** Replacing chaotic selection with **content-addressable retrieval over raw
embeddings** + PC-synchronization as a retrieval amplifier.

**Key findings (strict run `final_benchmark_v2.py`, public corpus, 6000 steps, 5 seeds, 95% CI):**
- **STS-Prog beats Transformer on PPL at a SMALLER parameter budget: 24.5 vs 46.6 (−47%)** at 900K vs 941K (matched D=92). Even against a somewhat smaller Transformer (D=88, 866K) — 24.5 vs 49.1.
- **The induction-retrieval advantage comes entirely from chaos/PC.** Plain content-addressable retrieval without PC-sync (`no-PC`, 17.3% at L16) does not beat the transformer (21.8%); adding PC-sync lifts it to 49.9%.
- The PPL win also comes from chaos/PC: `no-PC` 46.5 vs full STS-Prog 24.5 (both at 900K) — ~1.9× PPL improvement.
- All numbers are on a **public** corpus (TheAlgorithms/Python, MIT), reproducible off this machine (see TODO #3 and `SCIENCE_AUDIT.md`).

**Head-to-head vs Transformer (public corpus, 8 layers, 300 trials/distance × 5 seeds, fixed retrieval seed):**

### Parameter-matched benchmark v2 (strict run, `final_benchmark_v2.py`)

| Model | PPL mean±std | 95% CI | Params | Retr 16 | Retr 32 | Retr 64 | Retr 128 | Retr 256 |
|---|---|---|---|---|---|---|---|---|
| STS-Prog | 24.525±0.352 | [24.12, 24.93] | 900,353 | 49.9% (n=7500) | 51.9% (n=7500) | 41.3% (n=7500) | 42.0% (n=7500) | 37.9% (n=7500) |
| STS-Prog (no-PC) | 46.483±0.975 | [45.36, 47.60] | 900,353 | 17.3% (n=7500) | 15.3% (n=7500) | 16.7% (n=7500) | 16.0% (n=7500) | 15.2% (n=7500) |
| Transformer (D=88) | 49.132±1.526 | [47.38, 50.89] | 865,904 | 20.3% (n=7500) | 20.5% (n=7500) | 18.1% (n=7500) | 16.3% (n=7500) | 16.9% (n=7500) |
| Transformer-matched (D=92) | 46.620±1.492 | [44.91, 48.33] | 940,568 | 21.8% (n=7500) | 20.7% (n=7500) | 19.4% (n=7500) | 16.1% (n=7500) | 15.8% (n=7500) |

### Where the numbers come from (v2)

| Field | Value |
|---|---|
| commit | 9832a8672b014b15e70eb199c371a04e2136a441 |
| device | cuda |
| steps | 6000 |
| batch | 64 |
| lr | 0.0005 |
| warmup | 1000 |
| retrials | 300 |
| eval_seed | 12345 |
| protocol | FINAL_BENCHMARK.md (v2 rigorous) |
| seeds | [0, 1, 2, 3, 4] |

> **v2 methodology (why these numbers are trustworthy):**
> - Retrieval is computed on a **single fixed sample** (`eval_seed`), so across-seed variance = pure model variance, not a mix of model and sample variance (the v1 error, where the test sample changed per seed).
> - Transformer-matched is sized **no smaller** than STS by parameters (D=92, ~941K ≥ STS 900K): the STS win is conservative, against a larger opponent budget.
> - PPL is given as mean±std and 95% CI (t-distribution over the number of seeds). Retrieval = fraction of correct B predictions in the A→B pattern, averaged over all accepted trials (n_trials per seed, pooled across seeds; denominator in the JSON).
> - Results are written **incrementally** after each seed + `.pt` checkpoints, so a crash does not lose runs.

**Files:** `phase01/exp_vq/models_pc.py`, `final_benchmark_v2.py` (strict run), `prepare_public_corpus.py`, `render_docs_v2.py`, `rebuild_from_perseed.py`
**Details:** `phase01/exp_vq/README.md`, `SCIENCE_AUDIT.md`
**Status:** ✅ Confirmed by a strict 5-seed run (public corpus, 95% CI) — an attention alternative with lower PPL and better retrieval at comparable/smaller parameters.

---

## Repository structure

```
chaotic-llm/
├── README.md                              ← this file
├── .gitignore
│
├── phase01/                               ← all experiments
│   ├── archive/                           ← Phase 0–3, retired from active work:
│   │   ├── exp_morin_boost/               ←   order preserved (git sees as rename)
│   │   ├── exp_prior_prop/
│   │   └── exp_eye_selector/
│   ├── exp_01…exp_56/                     ← Phase 0–3: Arnold, chaos, old hypotheses
│   │   ├── experiment.py
│   │   └── results.json
│   ├── exp_vq/                            ← Phase 4–5: PC + STS-Prog (BREAKTHROUGH)
│   │   ├── README.md                      ← detailed description of the winner
│   │   ├── models_pc.py                   ← PurePCLM + STS-Prog architecture
│   │   ├── experiment_pc.py               ← training protocol + aux loss
│   │   ├── match_transformer.py           ← honest match vs transformer
│   │   ├── pc_probe*.py                   ← PC-sync probes
│   │   ├── diagnose_las*.py               ← selection diagnostics
│   │   ├── link_keeper_probe.py           ← Link Keeper probe (FAIL)
│   │   ├── night_sts_prog.py              ← night run on Stack
│   │   ├── night_transformer.py           ← night transformer match
│   │   ├── chat_sts_prog.py               ← inference/generation
│   │   ├── results_*.json                 ← all results
│   │   └── night_ckpt_*.pt                ← checkpoints
│   │
│   ├── corpus_train.txt                   ← small corpus (487K tokens)
│   ├── corpus_stack_train.txt             ← Stack (2GB, 973M tokens) — night runs only
│   ├── parametric_models.py               ← TransformerLM, auxiliary models
│   ├── chaos_lib.py                       ← chaotic map library
│   └── FINAL_REPORT.md                    ← Phase 0–3 report (historical)
│
├── experiments/                           ← Phase 0–3 toy-experiments (historical)
│
├── FINAL_REPORT.md                        ← historical
├── interactive_map.html                   ← visualization (historical)
└── [old files: chaos_lib.py, phase2.py, …] — historical, see Phase 0–3
```

---

## How to reproduce the win

```bash
cd phase01/exp_vq

# 1. Base sts_prog model (900K, small corpus)
python experiment_pc.py pc --driver sts_prog --k 1.2 --sync-steps 8 --layers 8 --d 192 --aux-w 0.5 --aux-mode multibead

# 2. Match vs transformer (8 layers, aligned protocol)
python match_transformer.py

# 3. Strict reproducible run (public corpus, 5 seeds, 95% CI) — see SCIENCE_AUDIT.md
python final_benchmark_v2.py --corpus ../corpus_public.txt --device cuda --steps 6000 --seeds 0,1,2,3,4 --retrials 300 --eval-seed 12345

# 4. Inference (text generation)
python chat_sts_prog.py --prompt "def fibonacci(n):" --steps 80 --temp 0.8
```

---

## What's next

### Immediate steps
- [ ] Training on a chat dataset (question→answer) — so the model can "talk"
- [ ] Integration with Hermes (local OpenAI-compatible server)
- [x] Repository cleanup: old experiments → `phase01/archive/`, single README, `.gitignore` for heavy data (`ru_chat.json`, 58 MB)
- [x] Honest final run (v1, `final_benchmark.py`, 3 seeds) + strict reproducible v2 (`final_benchmark_v2.py`, 5 seeds, public corpus, 95% CI): STS-Prog PPL 24.5 vs TF-matched 46.6; tables injected via `render_docs_v2.py` (see TODO #3 and `SCIENCE_AUDIT.md`)

### Long-term directions
- [ ] Scaling (100M+ parameters, 1B+ tokens)
- [ ] Comparison with transformer on long contexts (W≥1024)
- [ ] Optimization via Triton / torch.compile

---

## Key winner files

| File | What it is |
|---|---|
| `phase01/exp_vq/models_pc.py` | PurePCLM architecture + all selection modes |
| `phase01/exp_vq/experiment_pc.py` | Training protocol, aux loss, eval |
| `phase01/exp_vq/match_transformer.py` | Honest match vs transformer |
| `phase01/exp_vq/README.md` | Detailed description of the winner |
| `phase01/exp_vq/chat_sts_prog.py` | Inference/generation |

---

## Known gaps in the proofs (TODO)

An honest list of places where a claimed number is not backed by an artifact in the repo.
Nothing is hidden or deleted — this is a reminder that the run needs to be redone.

**1. `19.22` and `retrieval 47/40/35/35` — ✅ CLOSED by the final run.**
`phase01/exp_vq/results_pc.json` at the current HEAD contains
`mixer_ppl 23.991, retrieval 0.183/0.176/0.198/0.156` — this is an ablation run
`sts_prog_nopc`, not the winner (the same numbers appear in the attribution table).
The numbers 19.22 / 47% lived only in the README and in commit `32db903`.
**Now:** the final run `final_benchmark.py` (d=192, L=8, k=1.2, 3 seeds, 6000 steps)
gave STS-Prog PPL **17.94** (mixer 19.22 was unrealistically optimistic). Results
are laid out in `benchmark/*.json` and inserted into the table above via `render_docs.py`.
The old table (19.22 / 47%) was replaced with artifact-generated values.

**2. Different `k` in the reproduction commands.** Root README — `--k 1.2`,
`phase01/exp_vq/README.md` — `--k 1.8`. sigmoid(1.2) = 0.77 (the very k=0.77
from the fix description), sigmoid(1.8) = 0.86. One of the two variants does not
reproduce the claimed result. **The final run `final_benchmark.py`
uses `k_init=1.2`** (as in the root README) — the gap is closed for it.

**3. Reproducibility, parameter-match, and small-run statistics — ✅ CLOSED.**
The v1 final run (`final_benchmark.py`, 3 seeds) confirmed the direction but left three
gaps not in the list above:
- **(a) Private corpus.** `final_benchmark.py` reads `phase01/corpus_train.txt`, which
  is in `.gitignore` (leak of private sources) — results cannot be reproduced off
  this machine. Fix: `prepare_public_corpus.py` downloads the public code corpus
  **TheAlgorithms/Python** (MIT, pinned SHA `9391f546d6f8`) → `phase01/corpus_public.txt`;
  the v2 run (`final_benchmark_v2.py --corpus ...`) runs on it.
- **(b) Parameter-match was dishonest.** STS-Prog (900 353) had **+4%** over the transformer
  (865 904); the "matched" config took the smaller model, so the advantage stayed with STS.
  In v2 the matched-Transformer = **D=92 (~941K) ≥ STS 900K** — a conservative comparison
  (STS beats a model with a larger budget).
- **(c) Retrieval statistics wobbled across seeds.** In v1 the retrieval test was seeded with the model seed
  (`np.random.seed(seed)` inside `train_model`), so the TEST SAMPLE changed per seed,
  and the spread = a mix of model and sample variance. In v2 retrieval is computed on a
  **fixed sample** (`eval_seed`), plus 5 seeds and **95% CI** (t-distribution).
- **Status: ✅ CLOSED.** The `final_benchmark_v2.py` run is complete (5 seeds × 4 configurations × 6000 steps,
  retrieval 300 trials/distance) on the public corpus `corpus_public.txt`; tables with 95% CI are injected into the README
  via `render_docs_v2.py`. Gaps (a)–(c) are closed: new corpus, conservative parameter-match, and
  95% CI statistics are now reproducible off this machine (see `SCIENCE_AUDIT.md`).
