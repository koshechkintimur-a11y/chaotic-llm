"""exp35_confidence_gated_beta.py — β(c_h) closes the sparse-MLE gap for free.

exp32: sparse MLE on NL = 28.28 PPL vs KN 23.21 (9× cheaper memory, 87% of gain).
exp34: joint training breaks the mixer (mixer-only 4888 vs 62.7) — cleared.

"А что если": the gap comes from fixed β=0.9 trusting the memory equally
everywhere. But memory is reliable only on FREQUENT contexts (high c_h).
A confidence-gated β — β(c_h) = c_h/(c_h + k), one parameter, free (c_h
comes from the same lookup) — trusts memory less on rare contexts and lets
the (standalone-trained) mixer take over where memory is noisy.

This is NOT exp28's learned controller (AUC 0.6): it is a deterministic
function of the context's own evidence, not a predictor of y.

Compare (NL, WikiText-2, same positions as exp32):
  sparse MLE, fixed β=0.9:  28.28  (baseline)
  sparse MLE, β(c_h), k tuned:  ?
  KN, fixed β=0.9:          23.21  (target)
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
OUT = os.path.join(HERE, "exp35_confidence_beta")
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
MAX_TRAIN_CHARS = 4_000_000
N_TR, N_VA, N_TE = 60_000, 10_000, 12_000
BATCH = 1024


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_train.txt"), MAX_TRAIN_CHARS)
test_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_test.txt"))


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
print(f"contexts: {len(ctx_counts):,}")


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
model.load_state_dict(torch.load(os.path.join(HERE, "exp23_nl_beta", "nl_mixer.pt"), weights_only=True))
model.eval()

rng = np.random.default_rng(42)
maxstart_tr = len(train_ids) - W - 1
maxstart_te = len(test_ids) - W - 1
all_tr = np.sort(rng.choice(maxstart_tr, size=N_TR + N_VA, replace=False))
tr_starts, va_starts = all_tr[:N_TR], all_tr[N_TR:]
te_starts = np.sort(rng.choice(maxstart_te, size=N_TE, replace=False))


def extract(starts, seq):
    N = len(starts)
    logits_all = np.zeros((N, V), dtype=np.float32)
    ys = np.zeros(N, dtype=np.int64)
    with torch.no_grad():
        for s0 in range(0, N, BATCH):
            e = min(s0 + BATCH, N)
            X = np.zeros((e - s0, W), dtype=np.int64)
            for k, i in enumerate(starts[s0:e]):
                X[k] = seq[i:i + W]
                ys[s0 + k] = seq[i + W]
            logits_all[s0:e] = model(torch.tensor(X, dtype=torch.long)).numpy()
    return logits_all, ys


print("extract...")
logits_va, y_va = extract(va_starts, train_ids)
logits_te, y_te = extract(te_starts, test_ids)
lpmix_va = torch.log_softmax(torch.tensor(logits_va), -1).double().numpy()
lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()


def eval_gated(starts, seq, lpmix, ys, k_beta, beta_max=0.9):
    """β(c_h) = beta_max·c_h/(c_h + k). Memory = sparse MLE (observed continuations)."""
    N = len(ys)
    nll = np.zeros(N)
    for k in range(N):
        tpos = starts[k] + W
        ctx = tuple(seq[tpos - ORDER + 1:tpos])
        e = ctx_counts.get(ctx)
        pm = lpmix[k, ys[k]]
        if e:
            tot = sum(e.values())
            c = e.get(ys[k], 0)
            lp_mem = np.log(c / tot) if c > 0 else -1e9
            beta = beta_max * tot / (tot + k_beta)
        else:
            lp_mem = -1e9
            beta = 0.0
        if beta <= 0 or lp_mem <= -1e8:
            fused = pm
        else:
            fused = np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem)
        nll[k] = -fused
    return float(np.exp(np.mean(nll)))


def eval_fixed(starts, seq, lpmix, ys, beta):
    """sparse MLE with fixed β (baseline from exp32)."""
    N = len(ys)
    nll = np.zeros(N)
    for k in range(N):
        tpos = starts[k] + W
        ctx = tuple(seq[tpos - ORDER + 1:tpos])
        e = ctx_counts.get(ctx)
        pm = lpmix[k, ys[k]]
        if e:
            tot = sum(e.values())
            c = e.get(ys[k], 0)
            lp_mem = np.log(c / tot) if c > 0 else -1e9
        else:
            lp_mem = -1e9
        fused = np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + lp_mem)
        nll[k] = -fused
    return float(np.exp(np.mean(nll)))


print("\n=== val: β(c_h) gating ===")
best = (1e9, None)
for k_beta in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    for bmax in [0.7, 0.9, 1.0]:
        p = eval_gated(va_starts, train_ids, lpmix_va, y_va, k_beta, bmax)
        if p < best[0]:
            best = (p, (k_beta, bmax))
        print(f"  k={k_beta} βmax={bmax}: PPL {p:.3f}")
print(f"best: k={best[1][0]} βmax={best[1][1]} PPL {best[0]:.3f}")
k_b, bm = best[1]

print("\n=== baselines (val) ===")
for beta in [0.5, 0.9]:
    print(f"  fixed β={beta}: PPL {eval_fixed(va_starts, train_ids, lpmix_va, y_va, beta):.3f}")

print("\n=== test ===")
res = {}
p_fixed9 = eval_fixed(te_starts, test_ids, lpmix_te, y_te, 0.9)
p_gated = eval_gated(te_starts, test_ids, lpmix_te, y_te, k_b, bm)
res["fixed_beta09"] = p_fixed9
res["gated"] = p_gated
res["params"] = {"k": k_b, "beta_max": bm}
print(f"  sparse fixed β=0.9: PPL {p_fixed9:.3f}")
print(f"  sparse β(c_h):      PPL {p_gated:.3f}  (Δ {p_gated - p_fixed9:+.3f})")
print(f"  (KN target: 23.21)")

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)
