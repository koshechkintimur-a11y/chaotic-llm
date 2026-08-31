"""models_pc.py — PC-микшер: диссипативная Пекоры–Кэрролл динамика вместо
симметричного coupling. Построен на пробах pc_probe3 (sync_err=0 при k=1.2).

Ключевое отличие от exp52: coupling ОДНОНАПРАВЛЕННЫЙ + СЖИМАЮЩИЙ (α<1) —
создаёт устойчивое многообразие, к которому состояние синхронизируется.
Симметричный coupling (exp52) консервативен -> синхронизация невозможна.

Архитектура:
  embed + pos
  -> L блоков: Arnold-permute + диссипативный coupling + tanh
  -> PC-синхронизация readout к глобальному драйверу (mean)
  -> readout
"""
import math
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

try:
    from chaos_lib import permute_indices
except ImportError:
    def permute_indices(N, seed):
        rng = np.random.default_rng(seed)
        return rng.permutation(N)

W = 256


def count_params(m):
    return sum(p.numel() for p in m.parameters())


class PCDissipativeBlock(nn.Module):
    """Один блок: перестановка + однонаправленный сжимающий coupling + tanh.

    Схема coupling (Пекоры-Кэрролл, однонаправленная):
      even' = even + g * odd          # чётные тянутся к нечётным
      odd'  = α * odd + g * even'     # нечётные сжаты (α<1) и тянутся к even'
    α<1 -> диссипация -> устойчивое многообразие -> синхронизация.
    """
    def __init__(self, d, seed=0, alpha=0.9):
        super().__init__()
        self.d = d
        self.alpha = alpha
        self.sig = torch.tensor(permute_indices(W, seed), dtype=torch.long)
        self.inv = torch.argsort(self.sig)
        self.gate = nn.Parameter(torch.full((1,), 0.3))
        self.gate2 = nn.Parameter(torch.full((1,), 0.3))

    def forward(self, h):
        B = h.shape[0]
        # перестановка позиций (адреса)
        h = h[:, self.sig]
        # однонаправленный coupling по чёт/нечет
        e = h[:, 0::2]      # чётные
        o = h[:, 1::2]      # нечётные
        g = torch.sigmoid(self.gate) * 0.9 + 0.05
        g2 = torch.sigmoid(self.gate2) * 0.9 + 0.05
        e2 = e + g * o
        o2 = self.alpha * o + g2 * e2
        # собрать интерливом
        Wp = e2.shape[1]
        h2 = torch.stack([e2, o2], dim=2).reshape(B, 2 * Wp, self.d)
        if h2.shape[1] > h.shape[1]:
            h2 = h2[:, :h.shape[1]]
        elif h2.shape[1] < h.shape[1]:
            h2 = torch.cat([h2, h[:, h2.shape[1]:]], dim=1)
        # обратная перестановка
        h2 = h2[:, self.inv]
        return torch.tanh(h2)


class PCLM(nn.Module):
    """PC-микшер LM: диссипативная динамика + синхронизация readout к драйверу."""
    def __init__(self, vocab=VOCAB if 'VOCAB' in dir() else 512, d=128, layers=4, alpha=0.9):
        super().__init__()
        self.d = d
        self.layers = layers
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.blocks = nn.ModuleList(
            [PCDissipativeBlock(d, seed=i, alpha=alpha) for i in range(layers)])
        # PC-синхронизация readout к драйверу: k обучаемый (старт 1.2 из пробы)
        self.k = nn.Parameter(torch.tensor([1.2]))
        self.readout = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, vocab))

    def forward(self, x, return_aux=False):
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        # драйвер = глобальный контекст (mean по позициям)
        driver = h.mean(dim=1, keepdim=True)          # [B,1,d]
        k = torch.clamp(self.k, 0.0, 2.0)
        h_sync = h + k * (driver - h)                  # PC-связь, B шагов=1
        h_last = h_sync[:, -1, :]
        g = h.mean(dim=1)
        logits = self.readout(torch.cat([h_last, g], dim=-1))
        if return_aux:
            return logits, h_sync
        return logits


def build_pc_model(config, vocab=512, d=128):
    if config == "pc":
        return PCLM(vocab=vocab, d=d, layers=4)
    raise ValueError(f"unknown {config}")


if __name__ == "__main__":
    for cfg in ["pc"]:
        m = build_pc_model(cfg)
        x = torch.randint(0, 512, (2, W))
        y = m(x)
        print(f"{cfg}: params={count_params(m):,} out={tuple(y.shape)}")
