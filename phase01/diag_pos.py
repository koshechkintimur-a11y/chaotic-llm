"""Sanity: single-position eval exactly like exp18/22."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
W, Wl, bl, bg, D, V = 256, 64, 8, 4, 64, 512


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), 2_000_000)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=[], show_progress=False)
tok.train_from_iterator([train_text[i:i + 100000] for i in range(0, len(train_text), 100000)],
                        trainer=trainer)
test_ids = tok.encode(load_chars(os.path.join(HERE, "corpus_test.txt"))).ids
print("test tokens", len(test_ids))


class ChaoticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nw = W // Wl
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long) for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long) for t in range(1, bg + 1)}

    def _chaotic(self, h, sigmas, gates):
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even, odd = h[:, 0::2, :], h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(h.shape[0], h.shape[1], D)
        return h

    def forward(self, h):
        B, Wd, d = h.shape
        hw = h.view(B, self.Nw, Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l) for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        return loc.reshape(B, Wd, d) + glob.mean(dim=1, keepdim=True)


class ChaoticBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        self.block = ChaoticBlock()
        self.norm = nn.LayerNorm(D)

    def mix(self, x):
        return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))


class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ChaoticBase()
        self.readout = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1()
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                                 weights_only=True))
model.eval()

with torch.no_grad():
    for i in [0, 100, 1000, 10000, 50000, 200000, 500000]:
        ctx = test_ids[i:i + W]
        y = test_ids[i + W]
        logits = model(torch.tensor([ctx], dtype=torch.long))
        logp = torch.log_softmax(logits[0], -1)
        print(f"pos {i}: y={y} logp(y)={logp[y].item():.3f} top1={logp.argmax().item()} (y={y})")
