"""link_keeper_probe.py — Фаза A: проба Link Keeper на замороженной модели.

Сравнение трёх способов выбора драйвера на ПРАВИЛЬНОМ индукционном тесте:
  1. sts_emb (сканирование окна O(W), baseline)
  2. Link Keeper (lookup по архиву O(M·d), M≪W)
  3. Oracle (идеальный KEY+1, верхняя граница)

Правильный индукционный протокол (как в experiment_pc.py):
  - A→B на (i, i+1); второй A на j-1 (последний токен окна); окно [j-W, j).
  - Модель предсказывает следующий токен == B.
  - KEY (первый A) в окне на позиции W-L, B на W-L+1.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp_memory_selector"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiment import load_chars, make_bpe, MAX_TRAIN, VOCAB, W, D
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

train_text = load_chars(os.path.join(HERE, "..", "corpus_train.txt"), MAX_TRAIN)
test_text = load_chars(os.path.join(HERE, "..", "corpus_test.txt"))
tok = make_bpe(train_text)
V = tok.get_vocab_size()
test_ids = np.array(tok.encode(test_text).ids, dtype=np.int32)
print(f"V={V} test={len(test_ids)}")

from models_pc import build_pc_model, W
model = build_pc_model("pc", V, d=192, alpha=0.9, k_init=1.8, sync_steps=8, layers=8, driver_mode="sts_prog").to("cuda")
state = torch.load(os.path.join(HERE, "model_pc.pt"), map_location="cuda", weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
print(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params")


class LinkKeeper:
    """FIFO кольцевой буфер: ссылки (ключ t → значение t+1)."""
    def __init__(self, M):
        self.M = M
        self.keys, self.vals = [], []
        self.pos = 0
    def clear(self):
        self.keys.clear(); self.vals.clear(); self.pos = 0
    def add(self, key, val):
        if len(self.keys) < self.M:
            self.keys.append(key); self.vals.append(val)
        else:
            self.keys[self.pos] = key; self.vals[self.pos] = val
            self.pos = (self.pos + 1) % self.M
    def lookup(self, query, topk=1):
        if not self.keys:
            return None
        ks = torch.stack(self.keys); vs = torch.stack(self.vals)
        qn = query / (query.norm() + 1e-6)
        kn = ks / (ks.norm(dim=1, keepdim=True) + 1e-6)
        sim = (kn * qn).sum(dim=1)
        top_w, top_i = torch.topk(sim, min(topk, len(self.keys)))
        w = torch.softmax(top_w / 0.3, dim=0)
        return (w.unsqueeze(-1) * vs[top_i]).sum(dim=0)


def run_probe(model, test_ids, distances=(16, 64, 128, 256), n_trials=150, M=64):
    rng = np.random.default_rng(0)
    lk = LinkKeeper(M)
    res = {"sts": {}, "lk": {}, "oracle": {}}
    cos_all = {"sts": [], "lk": []}
    for L in distances:
        hits = {"sts": 0, "lk": 0, "oracle": 0}
        n = 0
        for _ in range(n_trials):
            found = False
            for _try in range(300):
                i = int(rng.integers(L + 2, len(test_ids) - L - 2))
                A = int(test_ids[i]); B = int(test_ids[i + 1])
                j = i + L
                if j < len(test_ids) and test_ids[j - 1] == A:
                    found = True
                    break
            if not found:
                continue
            window = test_ids[j - W:j]          # последний токен = A (второй)
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad():
                e = model.embed(X) + model.pos  # [1, W, d]
                # ===== ПРОГРЕССИВНЫЙ ЦИКЛ как в полном forward =====
                q0 = e[0, -model.nquery:, :].mean(dim=0)      # raw multi-query
                q = q0
                en = e[0] / (e[0].norm(dim=1, keepdim=True) + 1e-6)
                h = e
                for blk in model.blocks:
                    qn = q / (q.norm() + 1e-6)
                    sim = (en * qn.unsqueeze(0)).sum(dim=-1)  # [W]
                    sim[W - 8:] = -1e9
                    kk = min(model.topk, W - 8)
                    top_w, top_i = torch.topk(sim, kk)
                    w = torch.softmax(top_w / 0.3, dim=0)
                    top_next = torch.clamp(top_i + 1, 0, W - 2)
                    neigh = e[0, top_next]
                    driver_b = (w.unsqueeze(-1) * neigh).sum(dim=0)
                    h = blk(h, driver_b.unsqueeze(0).unsqueeze(0))
                    q = q0 + model.query_proj(h[0, -1]) * 0.5
                b_pos = W - L + 1
                ideal = e[0, b_pos]              # [d] идеальный KEY+1
                # ---- sts_emb: сканирование окна ----
                qn = q / (q.norm() + 1e-6)
                en = e[0] / (e[0].norm(dim=1, keepdim=True) + 1e-6)
                sim = (en * qn).sum(dim=1)
                sim[W - 8:] = -1e9
                sel = sim.argmax()
                driver_sts = e[0, torch.clamp(sel + 1, 0, W - 2)]
                # ---- Link Keeper: lookup по архиву ----
                lk.clear()
                for t in range(W - 1):
                    lk.add(e[0, t], e[0, t + 1])
                driver_lk = lk.lookup(q, topk=1)
                cos_all["sts"].append(torch.cosine_similarity(driver_sts.unsqueeze(0), ideal.unsqueeze(0)).item())
                if driver_lk is not None:
                    cos_all["lk"].append(torch.cosine_similarity(driver_lk.unsqueeze(0), ideal.unsqueeze(0)).item())
                # ---- синхронизация и предсказание ----
                k = torch.clamp(model.k, 0.0, 2.0)
                g = h.mean(dim=1)
                q0 = e[0, -model.nquery:, :].mean(dim=0)   # как в sts_prog forward
                for name, drv in (("sts", driver_sts), ("lk", driver_lk), ("oracle", ideal)):
                    if name == "lk" and driver_lk is None:
                        continue
                    h_sync = h[0, -1:]
                    for _ in range(model.sync_steps):
                        h_sync = h_sync + k * (drv.unsqueeze(0) - h_sync)
                    logits = model.readout3(torch.cat([h_sync[0], q0, g[0]], dim=-1))
                    if int(logits.argmax().item()) == B:
                        hits[name] += 1
                n += 1
        res["sts"][L] = hits["sts"] / max(1, n)
        res["lk"][L] = hits["lk"] / max(1, n)
        res["oracle"][L] = hits["oracle"] / max(1, n)
    print(f"\n=== Link Keeper проба (M={M}) ===")
    print(f"{'L':>6} {'sts_emb':>8} {'LK':>8} {'oracle':>8}")
    for L in distances:
        print(f"{L:>6} {res['sts'][L]:>8.3f} {res['lk'][L]:>8.3f} {res['oracle'][L]:>8.3f}")
    if cos_all["sts"]:
        print(f"\ncos vs ideal — sts: mu={np.mean(cos_all['sts']):.3f} std={np.std(cos_all['sts']):.3f}")
    if cos_all["lk"]:
        print(f"cos vs ideal — lk:  mu={np.mean(cos_all['lk']):.3f} std={np.std(cos_all['lk']):.3f}")
    return res


if __name__ == "__main__":
    for M in (32, 64, 128):
        run_probe(model, test_ids, M=M)