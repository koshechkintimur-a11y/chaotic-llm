"""exp36_betach_code_wallclock.py — β(c_h) on CODE + full architecture wall-clock.

exp35 showed confidence-gated β(c_h) on NL beats full KN (18.85 vs 23.21).
Question: does β(c_h) on CODE also beat/improve over fixed β?

Also: measure the FULL architecture wall-clock (mixer + sparse MLE + β(c_h)).
"""
import os
import sys
import time
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp36_code_betach")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
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


print("BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids

ctx_counts = defaultdict(dict)
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER + 1:i])
    w = train_ids[i]
    d_ = ctx_counts[ctx]
    d_[w] = d_.get(w, 0) + 1

# model
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
    def mix(self, x): return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))

class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ChaoticBase()
        self.readout = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))
    def forward(self, x):
        h = self.base.mix(x); gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))

model = ModelV1()
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"), weights_only=True))
model.eval()

rng = np.random.default_rng(42)
maxstart_te = len(test_ids) - W - 1
te_starts = np.sort(rng.choice(maxstart_te, size=N_EVAL, replace=False))

print("extract...")
logits_te, y_te = np.zeros((N_EVAL, V), dtype=np.float32), np.zeros(N_EVAL, dtype=np.int64)
with torch.no_grad():
    for s0 in range(0, N_EVAL, BATCH):
        e = min(s0+BATCH, N_EVAL)
        X = np.zeros((e-s0, W), dtype=np.int64)
        for k, i in enumerate(te_starts[s0:e]):
            X[k] = test_ids[i:i+W]; y_te[s0+k] = test_ids[i+W]
        logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long)).numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()
print("extracted")

# eval functions
def eval_gated(k_beta, beta_max=0.9):
    N = N_EVAL
    nll = np.zeros(N)
    for k in range(N):
        i = te_starts[k]
        pos = i + W
        ctx = tuple(test_ids[pos - ORDER + 1:pos])
        e = ctx_counts.get(ctx)
        pm = lpmix_te[k, y_te[k]]
        if e:
            tot = sum(e.values())
            c = e.get(int(y_te[k]), 0)
            lp_mem = np.log(c / tot) if c > 0 else -1e9
            beta = beta_max * tot / (tot + k_beta)
        else:
            lp_mem = -1e9
            beta = 0.0
        if beta <= 0 or lp_mem <= -1e8:
            nll[k] = -pm
        else:
            nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem)
    return float(np.exp(np.mean(nll)))


def eval_fixed(beta):
    N = N_EVAL
    nll = np.zeros(N)
    for k in range(N):
        i = te_starts[k]
        pos = i + W
        ctx = tuple(test_ids[pos - ORDER + 1:pos])
        e = ctx_counts.get(ctx)
        pm = lpmix_te[k, y_te[k]]
        lp_mem = -1e9
        if e:
            tot = sum(e.values())
            c = e.get(int(y_te[k]), 0)
            if c > 0:
                lp_mem = np.log(c / tot)
        nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem)
    return float(np.exp(np.mean(nll)))


print("\n=== test (code) ===")
res = {}
p_fix = eval_fixed(0.9)
print(f"  sparse fixed β=0.9:  PPL {p_fix:.3f}")
res["fixed_beta09"] = p_fix
best = (1e9, None)
for k_beta in [0.5, 1.0, 2.0, 5.0]:
    for bm in [0.9, 1.0]:
        p = eval_gated(k_beta, bm)
        print(f"  β(c_h) k={k_beta} βmax={bm}: PPL {p:.3f}")
        if p < best[0]:
            best = (p, (k_beta, bm))
print(f"  best: k={best[1][0]} βmax={best[1][1]} PPL {best[0]:.3f}")
res["gated"] = {"ppl": best[0], "k": best[1][0], "beta_max": best[1][1]}
print(f"  (KN target on code: 10.94)")

# ============ full architecture wall-clock ============
print("\n=== full architecture wall-clock (mixer + sparse MLE + β(c_h)) ===")
k_b, bm = best[1]

def run_once(gated):
    # mixer forward (batched)
    with torch.no_grad():
        X = np.zeros((N_EVAL, W), dtype=np.int64)
        for k, i in enumerate(te_starts):
            X[k] = test_ids[i:i + W]
        h = model.base.mix(torch.tensor(X, dtype=torch.long))
        gvec = h.mean(dim=1)
        logits = model.readout(torch.cat([h[:, -1, :], gvec], dim=-1))
    lpmix = torch.log_softmax(logits, -1).double().numpy()
    # memory
    for k in range(N_EVAL):
        i = te_starts[k]
        pos = i + W
        ctx = tuple(test_ids[pos - ORDER + 1:pos])
        e = ctx_counts.get(ctx)
        if e:
            _ = sum(e.values())
            _ = e.get(int(y_te[k]), 0)
    return lpmix


# warmup
for _ in range(3):
    run_once(True)
    run_once(False)

t_gated = []
for _ in range(5):
    t0 = time.perf_counter()
    run_once(True)
    t1 = time.perf_counter()
    t_gated.append(t1 - t0)
print(f"  mixer+sparse+β(c_h): mean {np.mean(t_gated)*1e3:.1f} ms "
      f"({np.mean(t_gated)/N_EVAL*1e6:.2f} µs/token)")

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)
