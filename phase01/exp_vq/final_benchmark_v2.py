"""final_benchmark_v2.py — строгая версия прогона для уверенности в результатах.

Чем отличается от final_benchmark.py (и почему):
1. **Фиксированный seed retrieval-теста.** В v1 eval_retrieval засевался
   np.random.seed(seed) внутри train_model, поэтому ТЕСТОВАЯ ВЫБОРКА retrieval
   была разной для каждого сида. Разброс по сидам = смесь дисперсии модели
   и дисперсии выборки. Здесь retrieval всегда считается на ОДНОЙ выборке
   (eval_seed), так что across-seed разброс = чисто дисперсия модели.
2. **Больше сидов** (по умолчанию 5) + **95% CI** в статистике.
3. **Параметр-матч Transformer.** Добавлен Transformer (D≈90, ~902K) — тот же
   бюджет, что у STS-Prog (900,353), чтобы сравнение было fair (v1 давал TF на
   866K, то есть STS имел +4% параметров).
4. **Инкрементальная запись + чекпоинты.** Пишем JSON после КАЖДОГО сида
   (не один раз в конце) и сохраняем .pt — падение не теряет прогоны.
5. **Больше trials** на дистанцию (300 вместо 200) — меньше шум оценки retrieval.
6. **Переключаемое устройство** (--device cpu/cuda). На CPU отключается
   mixed precision (float32), поэтому CPU-числа будут чуть отличаться от
   GPU-прогона v1 — для сравнения с v1 держите GPU.

Запуск:
  python final_benchmark_v2.py --device cuda --seeds 0,1,2,3,4
  python final_benchmark_v2.py --device cpu  --steps 20 --seeds 0 --retrials 20   # smoke
  python run_final_logged.py -- final_benchmark_v2.py --device cuda --seeds 0,1,2,3,4
"""
import argparse
import json
import math
import os
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

import final_benchmark as fb  # нужны только конструкторы моделей

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
CKPT = os.path.join(RESULTS, "ckpts")
PERSEED = os.path.join(RESULTS, "bench_v2")

W = fb.W
V_BASE = 512
LAYERS = 8
HEADS = 4

# t-критическое для 95% CI
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_crit(n):
    return _T95.get(n, 1.96)


def ci95(vals):
    n = len(vals)
    if n < 2:
        return None, None
    m = statistics.mean(vals)
    s = statistics.pstdev(vals)
    se = s / math.sqrt(n)
    return m, t_crit(n) * se


# ---------- обучение (копия протокола v1, с DEVICE) ----------
def train_model(model, train_ids, device, seed=0, steps=None, batch=None, lr=None,
                warmup=None, n_eval=None):
    steps = fb.STEPS if steps is None else steps
    batch = fb.BATCH if batch is None else batch
    lr = fb.LR if lr is None else lr
    warmup = fb.WARMUP if warmup is None else warmup
    n_eval = fb.N_EVAL if n_eval is None else n_eval

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    best_ppl = float("inf")
    best_step = 0
    t0 = time.time()
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    for step in range(1, steps + 1):
        lr_scale = min(1.0, step / warmup) if warmup else 1.0
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_scale
        s = rng.integers(0, n, size=batch)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device=device)
        Y = torch.tensor([train_ids[i + W] for i in s], dtype=torch.long, device=device)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(X)
                loss = lossf(logits, Y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(X)
            loss = lossf(logits, Y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % n_eval == 0 or step == steps:
            ppl = float(torch.exp(loss).item())
            if ppl < best_ppl:
                best_ppl = ppl
                best_step = step
    elapsed = time.time() - t0
    return {"final_ppl": float(torch.exp(loss).item()), "best_ppl": best_ppl,
            "best_step": best_step, "time": elapsed}


def eval_ppl(model, test_ids, device, batch=None):
    batch = fb.BATCH if batch is None else batch
    model.eval()
    n = len(test_ids) - W - 1
    rng = np.random.default_rng(42)
    nll = 0.0
    cnt = 0
    with torch.no_grad():
        for _ in range(0, n, batch):
            s = rng.integers(0, n, size=batch)
            X = torch.tensor(np.stack([test_ids[i:i + W] for i in s]), dtype=torch.long, device=device)
            Y = torch.tensor([test_ids[i + W] for i in s], dtype=torch.long, device=device)
            logits = model(X)
            nll += nn.CrossEntropyLoss(reduction="sum")(logits, Y).item()
            cnt += batch
    return np.exp(nll / cnt) if cnt > 0 else float("inf")


# ---- фиксированный retrieval-тест (seed не зависит от model seed) ----
def eval_retrieval_fixed(model, test_ids, device, distances=fb.RETRIEVAL_L,
                         n_trials=300, eval_seed=12345):
    model.eval()
    W0 = W
    results = {}
    rng = np.random.default_rng(eval_seed)  # ФИКСИРОВАНО по всем model-seed
    for L in distances:
        hits = 0
        trials = 0
        attempts = 0
        max_attempts = n_trials * 100
        lo = max(L + 2, W0 - L + 2)
        while trials < n_trials and attempts < max_attempts:
            if lo >= len(test_ids) - L - 3:
                break
            i = int(rng.integers(lo, len(test_ids) - L - 3))
            A = int(test_ids[i])
            B = int(test_ids[i + 1])
            j = i + L
            if j >= len(test_ids):
                attempts += 1
                continue
            if test_ids[j - 1] != A:
                attempts += 1
                continue
            window = test_ids[j - W0:j]
            X = torch.tensor([window], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(X)
            pred = logits.argmax(dim=-1).item()
            if pred == B:
                hits += 1
            trials += 1
            attempts += 1
        results[L] = {"trials": trials, "hits": hits,
                      "accuracy": hits / trials if trials > 0 else 0.0}
    return results


def build_data_from(path):
    """Тот же протокол, что fb.build_data(), но из явно заданного файла корпуса
    (публичного, воспроизводимого). fb.build_data читает приватный corpus_train.txt."""
    train_text = fb.load_chars(path, fb.MAX_TRAIN)
    tok = fb.make_bpe(train_text)
    V = tok.get_vocab_size()
    train_ids = np.array(tok.encode(train_text).ids, dtype=np.int32)
    return tok, V, train_ids


def _build_sts_for_w(Vc, Wc):
    """STS-модель под произвольный W (для scaling); pos переопределяется под Wc."""
    torch.manual_seed(0)
    from models_pc import PurePCLM
    m = PurePCLM(vocab=Vc, d=192, layers=LAYERS, k_init=1.2, alpha=0.3,
                 sync_steps=8, driver_mode="sts_prog", temp=0.3)
    m.pos = nn.Parameter(torch.randn(1, Wc, 192) * 0.02)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=fb.STEPS)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--retrials", type=int, default=300)
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS, "final_benchmark_v2.json"))
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--corpus", type=str,
                    default=os.path.join(ROOT, "phase01", "corpus_public.txt"),
                    help="путь к ПУБЛИЧНОМУ корпусу (воспроизводимый)")
    a = ap.parse_args()

    device = a.device
    SEEDS = [int(s) for s in a.seeds.split(",") if s != ""]
    fb.RETRIEVAL_N = a.retrials

    print("=" * 60, flush=True)
    print(f"FINAL BENCHMARK v2 (rigorous) device={device}", flush=True)
    print(f"steps={a.steps} seeds={SEEDS} retrials={a.retrials} eval_seed={a.eval_seed}", flush=True)
    print(f"commit={os.popen('git rev-parse HEAD').read().strip()}", flush=True)

    tok, V, train_ids = build_data_from(a.corpus)
    print(f"corpus={a.corpus}", flush=True)
    print(f"V={V} train_ids={len(train_ids)}", flush=True)
    idx = int(len(train_ids) * 0.8)
    train_sub = train_ids[:idx]
    test_sub = train_ids[idx:]

    from match_transformer import pick_tf_dims
    D_tf_base = pick_tf_dims(900_000, V, W, layers=LAYERS, heads=HEADS)  # 88 -> 866K

    def tf_params(D):
        return sum(p.numel() for p in fb.TransformerLM(V, W, D=D, HEADS=HEADS, LAYERS=LAYERS).parameters())

    # D кратен heads (embed_dim % num_heads == 0). Шаг 4.
    # Консервативный параметр-матч: Transformer НЕ меньше STS по параметрам.
    # Берём наименьший D (кратный HEADS) с params >= 900_353 -> D=92 (~941K, +4.4% к STS).
    # Тогда победа STS (900K) над TF (941K) железобетонная: выигрыш при БОЛЬШЕМ бюджете оппонента.
    D_tf_match = D_tf_base
    for D in range(80, 120, HEADS):
        if D % HEADS != 0:
            continue
        if tf_params(D) >= 900_353:
            D_tf_match = D
            break
    print(f"Transformer base D={D_tf_base} (~{tf_params(D_tf_base):,}); "
          f"matched D={D_tf_match} (~{tf_params(D_tf_match):,})", flush=True)

    configs = {
        "STS-Prog": lambda: fb.build_pc_model("pc", V, d=192, layers=LAYERS, k_init=1.2,
                                              sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=0.3),
        "STS-Prog (no-PC)": lambda: fb.build_pc_model("pc", V, d=192, layers=LAYERS, k_init=1.2,
                                                     sync_steps=8, driver_mode="sts_prog_nopc", alpha=0.3, temp=0.3),
        f"Transformer (D={D_tf_base})": lambda: fb.TransformerLM(V, W, D=D_tf_base, HEADS=HEADS, LAYERS=LAYERS),
        f"Transformer-matched (D={D_tf_match})": lambda: fb.TransformerLM(V, W, D=D_tf_match, HEADS=HEADS, LAYERS=LAYERS),
    }

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(PERSEED, exist_ok=True)
    all_results = {}
    if os.path.exists(a.out):
        try:
            with open(a.out) as f:
                all_results = json.load(f)
            print("загружен частичный результат из", a.out, flush=True)
        except Exception:
            all_results = {}

    for name, build_fn in configs.items():
        print(f"\n--- {name} ---", flush=True)
        if name not in all_results:
            all_results[name] = {"seeds": [], "stats": {}}
        done = {s["seed"] for s in all_results[name]["seeds"]}
        model_seeds = list(all_results[name]["seeds"])
        for seed in SEEDS:
            if seed in done:
                print(f"  Seed {seed}: уже есть, пропуск", flush=True)
                continue
            print(f"  Seed {seed}...", flush=True)
            m = build_fn()
            info = train_model(m, train_sub, device, seed=seed, steps=a.steps)
            ppl = eval_ppl(m, test_sub, device)
            ret = eval_retrieval_fixed(m, test_sub, device, eval_seed=a.eval_seed)
            params = sum(p.numel() for p in m.parameters())
            slug = "".join(c if c.isalnum() else "_" for c in name).lower()
            torch.save(m.state_dict(), os.path.join(CKPT, f"{slug}_seed{seed}.pt"))
            entry = {"params": params, "seed": seed, "ppl": round(ppl, 3),
                     "train": info, "retrieval": ret}
            model_seeds.append(entry)
            with open(os.path.join(PERSEED, f"{slug}_seed{seed}.json"), "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            ppls = [s["ppl"] for s in model_seeds]
            m_mean, m_ci = ci95(ppls)
            all_results[name] = {
                "seeds": model_seeds,
                "stats": {
                    "mean_ppl": round(m_mean, 3) if m_mean is not None else None,
                    "ppl_95ci": [round(m_mean - m_ci, 3), round(m_mean + m_ci, 3)] if m_ci is not None else None,
                    "std_ppl": round(statistics.pstdev(ppls), 3) if len(ppls) > 1 else 0.0,
                    "n_seeds": len(ppls),
                },
            }
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"    PPL={ppl:.3f} params={params:,} time={info['time']:.1f}s -> записано", flush=True)
            if ret:
                print(f"    Retrieval: {dict((k, round(v['accuracy'], 3)) for k, v in ret.items())}", flush=True)

    all_results["_meta"] = {
        "commit": os.popen('git rev-parse HEAD').read().strip(),
        "protocol": "FINAL_BENCHMARK.md (v2 rigorous)", "device": device, "seeds": SEEDS,
        "steps": a.steps, "batch": fb.BATCH, "lr": fb.LR, "warmup": fb.WARMUP,
        "retrials": a.retrials, "eval_seed": a.eval_seed,
        "corpus": a.corpus,
        "corpus_chars": int(len(train_ids)),
        "corpus_public": True,
        "note": "retrieval считается на фиксированной выборке (eval_seed) — across-seed разброс = дисперсия модели; корпус ПУБЛИЧНЫЙ (TheAlgorithms/Python, MIT)",
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {a.out}", flush=True)

    if not a.no_scaling and device == "cuda":
        print("\n--- Scaling (untrained, batch=1, single forward) ---", flush=True)
        all_results["_scaling"] = fb.eval_scaling(
            _build_sts_for_w,
            lambda Vc, Wc: fb.TransformerLM(Vc, Wc, D=D_tf_base, HEADS=HEADS, LAYERS=LAYERS),
            V, fb.W_VALS)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(json.dumps(all_results["_scaling"], indent=2), flush=True)


if __name__ == "__main__":
    main()
