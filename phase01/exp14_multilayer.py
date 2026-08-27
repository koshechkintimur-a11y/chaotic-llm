"""exp14_multilayer.py — Step 5a: cumulative depth gain for the chaotic mixer.

Train L=1, 2, 4 layers of the hierarchical chaotic mixer (each: local windows
8 blocks + global relay 4 blocks, residual connection) with the attention
readout + fixed beta-prior. Same data, steps, init -> does depth give a
cumulative gain like it does for transformers?

Saves checkpoints of all trained models for reuse (exp15, exp17).
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp14_multilayer")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORCE_CPU = True  # small chaotic ops are faster on CPU (kernel launch overhead)
if FORCE_CPU:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

VOCAB_SIZE = 512
W_LOCAL = 64
W = 256
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
BATCH = 64
MAX_TRAIN_BYTES = 2_000_000
ORDER = 2
BETA = 0.3
TRAIN_STEPS = 6000
LAYERS = [1, 2, 4]


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


train_text = load_chars(os.path.join(HERE, "corpus_train.txt"), MAX_TRAIN_BYTES)
test_text = load_chars(os.path.join(HERE, "corpus_test.txt"))


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)],
                            trainer=trainer)
    return tok


print("Training BPE...")
tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids
print(f"BPE vocab {V}, train tokens {len(train_ids):,}, test tokens {len(test_ids):,}")


# ============ beta corpus-prior ============
print("Building beta prior...")
prior = defaultdict(lambda: defaultdict(int))
for i in range(ORDER, len(train_ids)):
    prior[tuple(train_ids[i - ORDER:i])][train_ids[i]] += 1
prior = {k: dict(v) for k, v in prior.items()}
print(f"prior contexts: {len(prior):,}")

def prior_logp(ctx):
    for back in range(ORDER, 0, -1):
        table = prior.get(tuple(ctx[-back:]))
        if table:
            tot = sum(table.values())
            out = np.full(V, -1e9, dtype=np.float64)
            for t, c in table.items():
                out[t] = np.log(c / tot)
            return out
    return None


# ============ L-layer hierarchical chaotic mixer ============
class ChaoticBlock(nn.Module):
    """One hierarchical chaotic layer: local windows + global relay."""

    def __init__(self, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
                       for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, bg + 1)}

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, h):
        B, W, d = h.shape
        hw = h.view(B, self.Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        gvec = glob.mean(dim=1, keepdim=True)
        return loc.reshape(B, W, d) + gvec


class MultiLayerChaotic(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg, L):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.blocks = nn.ModuleList([ChaoticBlock(W, Wl, d, bl, bg) for _ in range(L)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = self.norm(h + blk(h))
        return h


class AttnReadout(nn.Module):
    def __init__(self, V, W, d):
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, h):
        B, W, d = h.shape
        q = self.q(h[:, -1:, :])
        k = self.kv(h)
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        out = self.proj(scores.softmax(-1) @ h)
        qh = h[:, -1:, :] + out
        return self.net(self.norm(qh).squeeze(1))


# ============ evaluate ============
def evaluate(mixer, readout, W, n_samples=20000, beta=BETA):
    mixer.eval()
    readout.eval()
    nll_m, nll_p, acc_m, acc_p = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            h = mixer(torch.tensor(ctx, dtype=torch.long, device=DEVICE).unsqueeze(0))
            logits = readout(h)
            logp = torch.log_softmax(logits[0], -1).cpu().numpy().astype(np.float64)
            nll_m.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc_m += 1
            cp = prior_logp(ctx)
            if cp is not None:
                logp_c = np.logaddexp(np.log1p(-beta) + logp, np.log(beta) + cp)
            else:
                logp_c = logp
            nll_p.append(-logp_c[y])
            if int(np.argmax(logp_c)) == y:
                acc_p += 1
            if len(nll_m) >= n_samples:
                break
    n = len(nll_m)
    mixer.train()
    readout.train()
    return {"n": n,
            "ppl_model": float(np.exp(np.mean(nll_m))),
            "ppl_prior": float(np.exp(np.mean(nll_p))),
            "ppl_gain": float(np.exp(np.mean(nll_m)) / np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_prior": acc_p / n}


def train(mixer, readout, steps=TRAIN_STEPS, tag=""):
    mixer.to(DEVICE)
    readout.to(DEVICE)
    params = list(mixer.parameters()) + list(readout.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    n = len(train_ids) - W - 1
    s = 0
    while s < steps:
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]),
                         dtype=torch.long, device=DEVICE)
        Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long, device=DEVICE)
        opt.zero_grad()
        h = mixer(X)
        loss = lossf(readout(h), Y)
        loss.backward()
        opt.step()
        s += 1
        if s % 4000 == 0:
            print(f"  [{tag} {s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    return loss.item()


results = {"V": V, "W": W, "W_local": W_LOCAL, "d": D_MODEL, "order": ORDER,
           "beta": BETA, "train_steps": TRAIN_STEPS, "train_tokens": len(train_ids),
           "test_tokens": len(test_ids)}

for L in LAYERS:
    tag = f"L{L}"
    mixer = MultiLayerChaotic(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL, L)
    readout = AttnReadout(V, W, D_MODEL)
    n_params = sum(p.numel() for p in mixer.parameters()) + \
               sum(p.numel() for p in readout.parameters())
    print(f"\n=== {tag} (params {n_params:,}) ===")
    ckpt_m = os.path.join(OUT, f"mixer_{L}.pt")
    ckpt_r = os.path.join(OUT, f"readout_{L}.pt")
    if os.path.exists(ckpt_m):
        mixer.load_state_dict(torch.load(ckpt_m))
        readout.load_state_dict(torch.load(ckpt_r))
        mixer.to(DEVICE)
        readout.to(DEVICE)
        print("loaded checkpoint")
    else:
        train(mixer, readout, tag=tag)
        torch.save(mixer.state_dict(), ckpt_m)
        torch.save(readout.state_dict(), ckpt_r)
    r = evaluate(mixer, readout, W)
    print(f"{tag}: {r}")
    results[f"L{L}"] = {**r, "params": n_params}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
