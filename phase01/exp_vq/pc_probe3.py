"""pc_probe3.py — финальная честная проба PC: синхронизация + СЕЛЕКЦИЯ.

Ключевые вопросы (после pc_probe: Арнольд консервативен -> нет синхронизации;
pc_probe2: Hénon расходится на торе):
  Q1. С диссипативной картой НА ТОРЕ ответ сходится к драйверу? (sync_err->0)
  Q2. Главный вопрос: для декодирования нужен ПЕРЕБОР всех драйверов (O(W),
      как attention) или синхронизация сама «находит» нужный?
      -> если нужен перебор, PC не решает проблему селекции — она остаётся.

Карта: x' = (a*x + y) mod 1  (растяжение по x)
       y' = (b*y + c) mod 1  (сжатие по y)  -> det = a*b < 1 -> диссипация
"""
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

W = 256
T = 80
N_TRIALS = 300
V = 512


def dissip_torus(p, a=1.9, b=0.5, c=0.11):
    """Диссипативная хаотическая карта на торе: растяжение по x, сжатие по y."""
    x = (a * p[..., 0] + p[..., 1]) % 1.0
    y = (b * p[..., 1] + c) % 1.0
    return torch.stack([x, y], dim=-1)


def pc_step(y, driver, k):
    fy = dissip_torus(y)
    return (fy + k * (driver - y)) % 1.0


def run_pc(ids, L, k):
    aux = (torch.arange(W, dtype=torch.float64) * 0.6180339887) % 1.0
    drivers = torch.stack([(ids.float() / V) % 1.0, aux.unsqueeze(0)], dim=-1)[0]
    hist = [drivers]
    s = drivers.clone()
    for t in range(T):
        s = dissip_torus(s)
        hist.append(s)
    hist = torch.stack(hist)

    y = torch.rand(2)
    driver_L = hist[:, L, :]
    for t in range(T):
        y = pc_step(y, driver_L[t], k)

    sync_err = (y - hist[-1, L]).abs().max().item()
    final = hist[-1]
    dists = (y.unsqueeze(0) - final).pow(2).sum(-1)
    est_L = int(dists.argmin().item())
    return sync_err, 1.0 if est_L == L else 0.0


def decode_requires_scan():
    """Демонстрация: без перебора драйверов ответ НЕ говорит «какой токен».
    Ответ — это точка на аттракторе; чтобы узнать идентичность, нужно
    сравнить со ВСЕМИ драйверами. Это O(W) = attention-подобная селекция."""
    return ("Для декодирования нужен перебор всех W драйверов (argmin по "
            "расстоянию). Синхронизация доставляет сигнал, но НЕ выбирает, "
            "какой токен важен — селекция остаётся внешней задачей (O(W)).")


rng = np.random.default_rng(0)
ids0 = torch.randint(0, V, (1, W))

print("=== Q1: синхронизация с диссипативной картой на торе ===")
for k in [0.1, 0.3, 0.6, 0.9, 1.2, 1.5]:
    errs = [run_pc(ids0, 64, k)[0] for _ in range(30)]
    print(f"  k={k:.1f}: sync_err={np.mean(errs):.4f}")

best_k = min([(np.mean([run_pc(ids0,64,k)[0] for _ in range(20)]), k)
              for k in [0.1,0.3,0.6,0.9,1.2,1.5]])[1]
print(f"\n=== Q2: декодирование с лучшим k={best_k} ===")
pc_acc = {}
for L in [16, 64, 128, 240]:
    accs = [run_pc(ids0, L, best_k)[1] for _ in range(N_TRIALS)]
    pc_acc[L] = float(np.mean(accs))
    print(f"  L={L}: decode={np.mean(accs):.3f}")

print(f"\nPC (k={best_k}): { {k: round(v,3) for k,v in pc_acc.items()} }")
print(f"\nВывод по селекции: {decode_requires_scan()}")
