"""_audit_ablation.py — проверка корректности абляции sts_prog_nopc.

Вопрос: в nopc ветка делает `h = h + driver * 0.0`. Доходит ли результат
СЕЛЕКЦИИ (driver) до logits? Если нет — абляция отключает не "хаос",
а весь контур селекции, и выводы об атрибуции некорректны.

Проверяем тремя способами:
  1) градиенты: есть ли ненулевой grad у блоков и у query_proj
  2) возмущение: меняем neigh (выбранные соседи) -> меняются ли logits
  3) подсчёт "живых" параметров
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.join(PHASE, "exp_memory_selector"))
sys.path.append(HERE)

from models_pc import build_pc_model, count_params

torch.manual_seed(0)
V, W = 512, 256
X = torch.randint(0, V, (4, W))

for mode in ("sts_prog", "sts_prog_nopc"):
    print("=" * 68)
    print(f"РЕЖИМ: {mode}")
    print("=" * 68)
    torch.manual_seed(0)
    m = build_pc_model("pc", vocab=V, d=192, layers=8, driver_mode=mode)
    total = count_params(m)
    m.train()
    logits = m(X)
    loss = logits.float().pow(2).mean()
    loss.backward()

    dead = 0
    live = 0
    rows = []
    for name, p in m.named_parameters():
        g = p.grad
        isnull = g is None
        gnorm = 0.0 if isnull else float(g.abs().sum())
        n = p.numel()
        if isnull or gnorm == 0.0:
            dead += n
        else:
            live += n
        if any(k in name for k in ("blocks.", "query_proj", "embed", "pos", "readout3")):
            rows.append((name, n, "НЕТ ГРАДИЕНТА" if (isnull or gnorm == 0.0) else f"{gnorm:.3e}"))
    print(f"  params всего        : {total:,}")
    print(f"  с ненулевым градиен.: {live:,}")
    print(f"  МЁРТВЫЕ (grad=0)    : {dead:,}")
    print("  ключевые тензоры:")
    for name, n, st in rows:
        print(f"    {name:28s} {n:>9,}  grad: {st}")

    # --- возмущение: подменяем выбранных соседей ---
    torch.manual_seed(0)
    m2 = build_pc_model("pc", vocab=V, d=192, layers=8, driver_mode=mode)
    m2.eval()
    with torch.no_grad():
        base = m2(X).clone()
    # ручная копия forward с подменой neigh на шум
    e = m2.embed(X) + m2.pos
    q0 = e[:, -m2.nquery:, :].mean(dim=1)
    q = q0
    en = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
    h = e
    torch.manual_seed(123)
    for blk in m2.blocks:
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        sim = (en * qn.unsqueeze(1)).sum(-1)
        sim[:, W - 8:] = -1e9
        kk = min(m2.topk, W - 8)
        top_w, top_i = torch.topk(sim, kk, dim=1)
        w = torch.softmax(top_w / m2.temp, dim=1)
        top_next = torch.clamp(top_i + 1, 0, W - 2)
        Bn = e.shape[0]
        idx = torch.arange(Bn).unsqueeze(1).expand(Bn, kk)
        neigh = e[idx, top_next]
        neigh = neigh + torch.randn_like(neigh) * 5.0        # СИЛЬНО искажаем выбор
        driver = (w.unsqueeze(-1) * neigh).sum(dim=1, keepdim=True)
        if mode == "sts_prog_nopc":
            h = h + driver * 0.0
        else:
            h = blk(h, driver)
        q = q0 + m2.query_proj(h[:, -1, :]) * 0.5
    h_last = h[:, -1, :]
    g = h.mean(dim=1)
    pert = m2.readout3(torch.cat([h_last, q0, g], dim=-1))
    delta = float((pert - base).abs().max())
    print(f"  возмущение выбранных соседей (σ=5): max|Δlogits| = {delta:.6f}")
    print(f"  => {'СЕЛЕКЦИЯ ВЛИЯЕТ на выход' if delta > 1e-6 else 'СЕЛЕКЦИЯ ОТКЛЮЧЕНА от выхода'}")
    print()
