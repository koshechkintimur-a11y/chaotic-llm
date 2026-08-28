"""eval_checkpoint_8000.py — compare scalar vs vector gates at step 8000.

Both checkpoints (exp41b scalar, exp42 vector) at ckpt_8000, same eval
protocol: mixer-only (control) + sparse β(c_h). Isolates gate-architecture
effect at identical training step.
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
W = 256
W_LOCAL = 64
D_MODEL = 512
BLOCKS_LOCAL = 16
BLOCKS_GLOBAL = 8
ORDER = 3
MAX_TRAIN_BYTES = 2_000_000
N_EVAL = 12000
EVAL_BATCH = 256


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


def make_bpe(text, vocab_size=512):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i+100000] for i in range(0, len(text), 100000)], trainer=trainer)
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


def build_model(vector_gates):
    class ChaoticBlock(nn.Module):
        def __init__(self, W, Wl, d, bl, bg):
            super().__init__()
            self.W, self.Wl, self.Nw = W, Wl, W // Wl
            self.d = d
            if vector_gates:
                self.gates_l = nn.Parameter(torch.zeros(bl, d))
                self.gates_g = nn.Parameter(torch.zeros(bg, d))
            else:
                self.gates_l = nn.Parameter(torch.zeros(bl))
                self.gates_g = nn.Parameter(torch.zeros(bg))
            self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long) for t in range(1, bl + 1)}
            self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long) for t in range(1, bg + 1)}

        def _chaotic(self, h, sigmas, gates):
            B, N, d = h.shape
            for t in range(1, len(gates) + 1):
                h = h[:, sigmas[t].to(h.device), :]
                g = torch.sigmoid(gates[t - 1])
                even, odd = h[:, 0::2, :], h[:, 1::2, :]
                h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
            return h

        def forward(self, h):
            B, W, d = h.shape
            hw = h.view(B, self.Nw, self.Wl, d)
            loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l) for wi in range(self.Nw)], dim=1)
            glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
            return loc.reshape(B, W, d) + glob.mean(dim=1, keepdim=True)

    class ChaoticBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, D_MODEL)
            self.pos = nn.Parameter(torch.randn(1, W, D_MODEL) * 0.02)
            self.block = ChaoticBlock(W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
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

    return ModelV1()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(42)
maxstart_te = len(test_ids) - W - 1
te_starts = np.sort(rng.choice(maxstart_te, size=N_EVAL, replace=False))

results = {}
for name, ckpt in [("scalar (exp41b)", "exp41b_rerun_diagnostics/ckpt_8000.pt"),
                   ("vector (exp42)", "exp42_vector_gates/ckpt_8000.pt")]:
    vector = "vector" in name
    model = build_model(vector).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(HERE, ckpt), weights_only=True))
    model.eval()
    print(f"\n=== {name} @ step 8000 ===")
    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=np.int64)
    with torch.no_grad():
        for s0 in range(0, N_EVAL, EVAL_BATCH):
            e = min(s0 + EVAL_BATCH, N_EVAL)
            X = np.zeros((e - s0, W), dtype=np.int64)
            for k, i in enumerate(te_starts[s0:e]):
                X[k] = test_ids[i:i + W]
                y_te[s0 + k] = test_ids[i + W]
            logits_te[s0:e] = model(torch.tensor(X, dtype=torch.long, device=DEVICE)).cpu().numpy()
    lpmix_te = torch.log_softmax(torch.tensor(logits_te), -1).double().numpy()
    mixer_only = float(np.exp(np.mean([-lpmix_te[k, y_te[k]] for k in range(N_EVAL)])))
    print(f"  mixer-only: PPL {mixer_only:.3f}")

    gated = {}
    for k_beta in [0.5, 1.0, 2.0]:
        nll = np.zeros(N_EVAL)
        for k in range(N_EVAL):
            i = te_starts[k]
            pos = i + W
            ctx = tuple(test_ids[pos - ORDER + 1:pos])
            e = ctx_counts.get(ctx)
            pm = lpmix_te[k, y_te[k]]
            if e:
                tot = sum(e.values())
                c = e.get(int(y_te[k]), 0)
                if c > 0:
                    beta = tot / (tot + k_beta)
                    nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + np.log(c / tot))
                    continue
            nll[k] = -pm
        p = float(np.exp(np.mean(nll)))
        gated[k_beta] = p
        print(f"  +sparse k={k_beta}: PPL {p:.3f}")
    results[name] = {"mixer_only": mixer_only, "gated": gated}

json.dump(results, open(os.path.join(HERE, "exp42_vector_gates", "compare_step8000.json"), "w"), indent=2)
print("\ncomparison saved")
