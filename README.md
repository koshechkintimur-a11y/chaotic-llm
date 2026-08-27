# ChaoticLLM — Controlled Reversible Chaotic Dynamics as an LLM Architecture

Исследовательский проект: можно ли использовать **обратимую хаотическую динамику**
(карта кота Арнольда, $A = \begin{pmatrix}1&1\\1&2\end{pmatrix}$ на $\mathbb{Z}_N^2$)
как вычислительный механизм для языковой модели — вместо части pairwise attention?

> **Главный вопрос:** Can controlled reversible chaotic dynamics replace part of
> pairwise attention? И если да — какова реальная вычислительная сложность?

**Краткий ответ:** условно да — но не как «хаос вычисляет», а как
**«дешёвый пропозер-перемешиватель + внешний селектор»**. Полная архитектура
(иерархический хаос-миксер + attention-readout + β-корпусный приор) на реальном
коде даёт PPL 11.24 / top-1 43.0% при ~30× более дешёвом перемешивании, чем
attention. Ограничения зафиксированы честно (см. Failure modes).

---

## Структура репозитория

```
chaotic-llm/
├── README.md                  ← этот файл
├── FINAL_REPORT.md            ← Phase 0–3: математика карты, toy-трансформер, ablations
├── chaos_lib.py               ← общая библиотека: карта Арнольда, метрики, FLOPs
├── chaotic_torch.py           ← PyTorch-модели (ChaoticMixer, GRU, Local/Full Attn, MLP)
├── toy_data.py                ← задачи A–D (associative recall, long-range, retrieval)
├── train_models.py            ← обучение всех моделей на задачах A–D
├── phase2.py                  ← GSF, adaptive depth, error controller, reversibility
├── phase3.py                  ← scaling, attention comparison, ablation
│
├── experiments/               ← Phase 0–3 (exp_00 … exp_13)
│   └── exp_XX/  README.md + results.json + plots/
│
└── phase01/                   ← контраргументная фаза + синтез (exp01 … exp17)
    ├── FINAL_REPORT.md        ← Phase 0.1: ответ на главный вопрос (C — EQUIVALENT)
    ├── FINAL_REPORT_v2.md     ← финальный синтез: архитектура v0.2
    ├── build_corpus.py        ← сбор корпуса из локальных проектов (НЕ коммитится)
    ├── expXX_*.py             ← каждый эксперимент: полный скрипт
    └── expXX_*/  README.md + results.json
```

> ⚠️ **corpus_train.txt / corpus_test.txt в git не попадают** — они построены из
> приватных проектов (`Desktop/03_Проекты`). Воспроизводимость: `python build_corpus.py`.

---

## Две фазы исследования

### Phase 0–3 (`experiments/`) — игрушечные задачи, обучение

| Эксперимент | Вопрос | Вердикт |
|---|---|---|
| exp_00 reversibility | $F^{-1}(F(X))=X$ для N=8..1024 | ✅ обратимость |
| exp_01 period | период T(N) карты | ✅ T(2^k)=3·2^{k-2}, дикая немонотонность |
| exp_02 mixing | хаос vs случайная/фикс. перестановка vs линейная | ✅ хаос — хороший перемешиватель |
| exp_03 information spread | сколько токенов затронуто | ⚠️ чистая пермутация НЕ распространяет сигнал |
| exp_04 token communication | retrieval через хаос | ⚠️ нужен combine |
| exp_05 long_range | accuracy(L), L=2..512 | ❌ хаос без иерархии не тянет long-range |
| exp_06 GSF | может ли контроллер направить хаос | ❌ GSF не даёт селективность |
| exp_07 adaptive depth | остановка по качеству | ✅ снижает compute |
| exp_08 error controller | коррекция траектории | ⚠️ работает, но дорого |
| exp_09 reversibility under control | сохраняется ли обратимость | ❌ управление ломает (0.36 vs 1e-6) |
| exp_10 toy transformer | обучение на задачах A–D | ⚠️ шанс без селектора |
| exp_11 scaling | compute(N), accuracy(N) | ⚠️ |
| exp_12 attention comparison | хаос vs attention на retrieval | ✅ ChaoticAttnReadout 0.92 при 9.3× дешевле |
| exp_13 ablation | убрать компоненты | ✅ перестановка без сцепления = шанс; GSF = 0 |

**Ключевые находки Phase 0–3:**
- Карта Арнольда — **обратимая пермутация**, она НЕ смешивает значения.
  Для коммуникации токенов обязательно сцепление (coupling).
- GSF-гейты — не селективны (0.189 vs 0.190).
- **Победа**: ChaoticAttnReadout 91.5% на retrieval при 3.66e4 FLOPs против
  3.40e5 у FullAttn (в ~9.3× дешевле).

### Phase 0.1 (`phase01/`) — контраргументы + синтез

| Эксперимент | Вопрос | Вердикт |
|---|---|---|
| exp01 addressable iteration | замкнутая форма A^t через Фибоначчи | ✅ адресация O(log t); AND не выразим |
| exp02 chaotic router | может ли хаос маршрутизировать | ❌ орбита = 0.75/N пространства |
| exp03 computation | вычисляет ли хаос | ❌ readout(A^t x) = readout(x) |
| exp04_05 error locality | локальность ошибок | ❌ семантической локальности нет |
| exp06 recurrence equivalence | хаос = RNN/hash | ⚠️ одна карта = O(N) орбит |
| exp07 poincare recurrence | возвращение Пуанкаре | ✅ точный возврат через T(N) |
| exp08 phantom filter | β-приор из morin-filter | ✅ **β — недостающий параметр** |
| exp09 chaotic LM | char-LM на реальном коде | ✅ хаос+β: PPL 4.94, 60.3% (2.3 п.п. от attention) |
| exp10 hierarchical LM | иерархия для long-range | ✅ PPL 42.9 → 8.96 (4.79×) при контексте 64→256 |
| exp11 BPE scaling | BPE-уровень, равный бюджет | ✅ паритет 15% PPL держится |
| exp12 full architecture | хаос + β-приор | ✅ 13.5 PPL / 41.4% (14% от attention) |
| exp13 two selectors | + attention-readout | ✅ **PPL 23.7 vs 25.7 у трансформера** (12K шагов) |
| exp14 multilayer | глубина L=1,2,4 | ❌ нет кумулятивного выигрыша (нет весов на слой) |
| exp15 learnable beta | β как параметр | ✅ **β=0.972 → PPL 11.24 / 43.0%** |
| exp16 GPU latency | W=512..2048 на RTX 3060 | ❌ wall-clock 76-225× медленнее (Python vs cuBLAS) |
| exp17 kNN-LM vs β | равная память | ✅ **β-приор в 1.84× лучше kNN** при равной памяти |

---

## Итоговая архитектура v0.2

```
Input (BPE-токены)
  → Embedding + Position
  → Hierarchical Chaotic Mixer (1 слой):
      локальные окна 64: 8×(permute_Arnold + coupling)      [O(W log W)]
      глобальное реле 4: 4×(permute_Arnold + coupling)
  → Attention Readout (запрос по всем W, контентная селекция) [O(W·d)]
  → Gate: β-микстура с корпусным приором (β = 0.972)          [O(1)]
  → Logits
```

**Рабочая гипотеза (подтверждена на реальном коде, BPE-512):**
- **Хаос = дешёвый пропозер** (обратимое перемешивание O(W log W), не обучаемое,
  ~12 параметров на блок).
- **Селектор = приор/readout** (корпусная статистика + контентная селекция).
- Вместе ≈ attention в пределах ~14% PPL при ~30× дешевле перемешивании.

**Лучшие результаты:**

| Метрика | Значение | Против |
|---|---|---|
| PPL (BPE-512, код) | **11.24** (хаос + β=0.972) | attention: 11.9–22.4 |
| top-1 | **43.0%** | attention: 42.4% |
| Retrieval (Phase 3) | 0.92 accuracy | 9.3× дешевле attention |
| Long-range | PPL 4.79× лучше при контексте 64→256 | иерархия |
| Перемешивание | O(W log W) vs O(W²) | ~30× (по FLOPs) |

---

## Failure modes (зафиксированы честно)

1. **Чистый хаос без селектора = шанс.** Пермутация не создаёт информацию.
2. **GSF-гейты не селективны** — только глобальные коэффициенты.
3. **Управление ломает обратимость** (exp09): 0.36 vs 1e-6.
4. **Глубина не помогает** (exp14): у хаос-блоков нет весов, L2≈L1, L4 хуже.
   Нужны пер-слойные проекции.
5. **GPU wall-clock** (exp16): attention (fused cuBLAS) в 76–225× быстрее при
   W ≤ 2048. Выигрыш хаоса проявится при W > 100K или с fused CUDA-ядрами.
6. **Контекстный β дивергирует** (exp15B) — нужен clamp.

---

## Воспроизводимость

**Зависимости:** `pip install torch numpy matplotlib tokenizers`

**Корпус:** соберите из своих проектов:
```bash
python phase01/build_corpus.py   # читает Desktop/03_Проекты, пишет corpus_train/test.txt
```

**Запуск отдельных экспериментов:**
```bash
# Phase 0–3
python experiments/exp_02_mixing/experiment.py
python train_models.py --task A
# Phase 0.1
python phase01/exp10_hierarchical_lm.py
python phase01/exp17_knn_lm.py
```

Каждый `expXX` содержит: `README.md` (гипотеза, setup, результаты, интерпретация,
failure modes, вывод), `results.json` (сырые метрики), местами `plots/`.

**GPU:** exp16 требует CUDA-сборку torch (`pip install torch --index-url
https://download.pytorch.org/whl/cu121`); при наличии nvrtc-расхождения DLL —
подложить `nvrtc64_121_0.dll`.

---

## Автор и контекст

Исследование — часть проекта ChaoticLLM (репозиторий пользователя). Вдохновлено
гипотезой, что обратимая хаотическая динамика может быть вычислительным
механизмом LLM; синтез опирается на β-корпусный приор (Jelinek-Mercer/kNN-LM,
идея из фильтра фантомов morin-filter).
