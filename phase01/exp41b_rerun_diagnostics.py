"""exp41b_rerun_diagnostics.py — d=512 rerun with full diagnostics protocol.

Colleague-reviewed protocol:
  - lr=5e-4, warmup 1000 (linear), cosine decay to 16K steps
  - grad-norm logging every 500 steps (total pre-clip, embed, readout)
  - reversibility error ||F^-1(F(X))-X|| on a val batch every 500 steps
  - checkpoints every 2000 steps
  - control eval WITHOUT memory (β=0) — mixer-only is the compute signal
    (memory masks the mixer contribution at V=512)

Discriminates: H1 optimizer mismatch vs H2 gradient attenuation through
24 permutations.
"""
import os
import sys
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp41b_rerun_diagnostics")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
D_MODEL = 512
BLOCKS_LOCAL = 16
BLOCKS_GLOBAL = 8
ORDER = 3
MAX_TRAIN_BYTES = 2_000_000
BATCH = 64
TRAIN_STEPS = 16000
LR = 5e-4
WARMUP = 1000
N_EVAL = 12000
EVAL_BATCH = 256


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


class ChaoticBlock(nn.Module):
    def __init__(self, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
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
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.block = ChaoticBlock(W, Wl, d, bl, bg)
        self.norm = nn.LayerNorm(d)

    def mix(self, x):
        return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))

    def reversibility_error(self, x):
        """||F^-1(F(X)) - X|| on mixed representations (analytic inverse of
        permute + coupling stages)."""
        with torch.no_grad():
            h = self.embed(x) + self.pos
            # forward through local blocks (one window) for a cheap probe
            hw = h[:, :W_LOCAL, :].unsqueeze(1)  # (B, 1, Wl, d) as one window
            b = self.block
            g = torch.sigmoid(b.gates_l)
            # forward: permute then couple, per block
            cur = hw[:, 0, :, :]
            for t in range(1, len(b.gates_l) + 1):
                cur = cur[b._sig_l[t].to(cur.device), :]
                gt = g[t - 1]
                even, odd = cur[0::2, :], cur[1::2, :]
                cur = torch.cat([even + gt * odd, odd], dim=0)
            # inverse
            for t in range(len(b.gates_l), 0, -1):
                gt = g[t - 1]
                even_c, odd_c = cur[0::2, :], cur[1::2, :]
                even = even_c - gt * odd_c
                cur = torch.cat([even, odd_c], dim=0)
                inv_idx = torch.argsort(b._sig_l[t].to(cur.device))
                cur = cur[inv_idx, :]
            err = (cur - hw[0, 0, :, :]).abs().max().item()
        return err


class ModelV1(nn.Module):
    def __init__(self, base, V):
        super().__init__()
        self.base = base
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base = ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
model = ModelV1(base, V).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}  device: {DEVICE}")

# schedule: linear warmup -> cosine
opt = torch.optim.Adam(model.parameters(), lr=LR)
def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    p = (step - WARMUP) / max(1, TRAIN_STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * p))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
lossf = nn.CrossEntropyLoss()

n = len(train_ids) - W - 1
t0 = time.time()
logs = []
print("training (16K steps, lr 5e-4, warmup+cosine)...")
for s in range(TRAIN_STEPS):
    bi = np.random.randint(0, n, size=BATCH)
    X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long, device=DEVICE)
    Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long, device=DEVICE)
    opt.zero_grad()
    loss = lossf(model(X), Y)
    loss.backward()
    # gradient diagnostics BEFORE clipping
    if s % 500 == 0:
        g_total = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
        g_embed = model.base.embed.weight.grad.norm().item() if model.base.embed.weight.grad is not None else 0.0
        g_readout = model.readout[-1].weight.grad.norm().item() if model.readout[-1].weight.grad is not None else 0.0
        rev = model.base.reversibility_error(X)
        logs.append({"step": s, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                     "grad_total": g_total, "grad_embed": g_embed,
                     "grad_readout": g_readout, "reversibility_err": rev})
        print(f"  [{s:,}] loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e} "
              f"|g|={g_total:.3f} (emb {g_embed:.3f}, ro {g_readout:.3f}) rev={rev:.2e} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if s % 2000 == 0 and s > 0:
        torch.save(model.state_dict(), os.path.join(OUT, f"ckpt_{s}.pt"))

torch.save(model.state_dict(), os.path.join(OUT, "mixer_d512_v2.pt"))
json.dump(logs, open(os.path.join(OUT, "train_logs.json"), "w"), indent=2)
print(f"trained + saved ({time.time()-t0:.0f}s)")

# ============ eval: control (no memory) + fused + KN ============
model.eval()
rng = np.random.default_rng(42)
maxstart_te = len(test_ids) - W - 1
te_starts = np.sort(rng.choice(maxstart_te, size=N_EVAL, replace=False))

logits_te, y_te = np.zeros((N_EVAL, V), dtype=np.float32), np.zeros(N_EVAL, dtype=np.int64)
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
print(f"\nCONTROL (β=0, mixer only): PPL {mixer_only:.3f}")
print(f"  [d=64: 32.78 | d=256: 16.73 | d=512 failed run: 87.72]")


def eval_gated(k_beta):
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
            if c > 0:
                beta = tot / (tot + k_beta)
                nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + np.log(c / tot))
                continue
        nll[k] = -pm
    return float(np.exp(np.mean(nll)))


print("  +sparse β(c_h):")
gated = {}
for k_beta in [0.5, 1.0, 2.0, 5.0]:
    p = eval_gated(k_beta)
    gated[k_beta] = p
    print(f"    k={k_beta}: PPL {p:.3f}")

json.dump({"params": n_params, "mixer_only": mixer_only, "gated": gated},
          open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("saved", OUT)
