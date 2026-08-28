"""exp38_vocab2048_betach.py — does β(c_h) + sparse MLE hold at BPE-2048?

exp26 showed full KN at BPE-2048: PPL 20.18 (mixer 77.6, memory 8.9 MB).
Question: does the v0.7 memory (sparse MLE + β(c_h)) also beat KN at 4×
vocabulary? Tests the scaling claim of the architecture.
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp38_vocab2048_betach")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 2048
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
ORDER = 3
MAX_TRAIN_BYTES = 2_000_000
N_EVAL = 12000
BATCH = 1024


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


print("BPE-2048...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}")

# sparse MLE memory
ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1
print(f"contexts: {len(ctx_counts):,}")


# model (V2048)
class ChaoticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nw = W // W_LOCAL
        self.gates_l = nn.Parameter(torch.zeros(BLOCKS_LOCAL))
        self.gates_g = nn.Parameter(torch.zeros(BLOCKS_GLOBAL))
        self._sig_l = {t: torch.as_tensor(permute_indices(W_LOCAL, t), dtype=torch.long) for t in range(1, BLOCKS_LOCAL + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long) for t in range(1, BLOCKS_GLOBAL + 1)}

    def _chaotic(self, h, sigmas, gates):
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even, odd = h[:, 0::2, :], h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(h.shape[0], h.shape[1], D_MODEL)
        return h

    def forward(self, h):
        B, Wd, d = h.shape
        hw = h.view(B, self.Nw, W_LOCAL, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l) for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        return loc.reshape(B, Wd, d) + glob.mean(dim=1, keepdim=True)


class ChaoticBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, W, D_MODEL) * 0.02)
        self.block = ChaoticBlock()
        self.norm = nn.LayerNorm(D_MODEL)

    def mix(self, x):
        return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))


class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ChaoticBase()
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


model = ModelV1()
model.load_state_dict(torch.load(os.path.join(HERE, "exp26_hash_table_vocab", "mixer_v2048.pt"),
                                 weights_only=True))
model.eval()
print("mixer v2048 loaded")

rng = np.random.default_rng(42)
maxstart_te = len(test_ids) - W - 1
te_starts = np.sort(rng.choice(maxstart_te, size=N_EVAL, replace=False))

print("extract...")
logits_te, y_te = np.zeros((N_EVAL, V), dtype=np.float32), np.zeros(N_EVAL, dtype=np.int64)
with torch.no_grad():
    for s0 in range(0, N_EVAL, BATCH):
        e = min(s0 + BATCH, N_EVAL)
        X = np.zeros((e - s0, W), dtype=np.int64)
        for k, i in enumerate(te_starts[s0:e]):
            X[k] = test_ids[i:i + W]
            y_te[s0 + k] = test_ids[i + W]
        logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long)).numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()


def eval_mem(starts, seq, lpmix, ys, kind, k_beta=0.5, beta_max=1.0, beta_fixed=0.9):
    N = len(ys)
    nll = np.zeros(N)
    for k in range(N):
        i = starts[k]
        pos = i + W
        ctx = tuple(seq[pos - ORDER + 1:pos])
        e = ctx_counts.get(ctx)
        pm = lpmix[k, ys[k]]
        if e:
            tot = sum(e.values())
            c = e.get(int(ys[k]), 0)
            lp_mem = np.log(c / tot) if c > 0 else -1e9
            if kind == "gated":
                beta = beta_max * tot / (tot + k_beta)
            else:
                beta = beta_fixed
        else:
            lp_mem = -1e9
            beta = 0.0 if kind == "gated" else beta_fixed
        nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem)
    return float(np.exp(np.mean(nll)))


print("\n=== test (BPE-2048) ===")
res = {}
p_mix = float(np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in range(N_EVAL)])))
res["mixer_only"] = p_mix
print(f"  mixer-only:        PPL {p_mix:.3f}  (exp26: 77.6)")

p_fix = eval_mem(te_starts, test_ids, lpmix_te, y_te, "fixed")
res["sparse_fixed_beta09"] = p_fix
print(f"  sparse fixed β=0.9: PPL {p_fix:.3f}")

for k_beta in [0.5, 1.0, 2.0, 5.0, 10.0]:
    p = eval_mem(te_starts, test_ids, lpmix_te, y_te, "gated", k_beta=k_beta)
    print(f"  sparse β(c_h) k={k_beta}: PPL {p:.3f}")
    res[f"sparse_gated_k{k_beta}"] = p

print(f"  (exp26 KN target: 20.18)")
print(f"  (per-token: KN {20.18/2048:.4f} vs gated {min(v for k,v in res.items() if 'gated' in k)/2048:.4f})")

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)
