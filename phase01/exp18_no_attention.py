"""exp18_no_attention.py — Step 6: remove attention-readout entirely.

Replace the attention readout with a selector that uses the dynamics of the
space itself.

Variants (NO attention anywhere):
  V1: hierarchical chaotic mixer + LOCAL readout (query position only) + beta
  V2: hierarchical chaotic mixer + QUERY-ANCHORED dynamics (query identity
      re-injected each anchor round, context mixed around it) + local readout
      + beta  — "selection via dynamics": the dynamics shapes the query's
      context summary; readout is a local MLP.

Baselines (for comparison, same 6000-step budget as exp14 L1):
  L1+attn-readout (exp14): PPL 32.5 / +beta 13.7 / 40.6%
  AttnLM full: PPL ~22-25
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
OUT = os.path.join(HERE, "exp18_no_attention")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

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
ANCHOR_ROUNDS = 2


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


# ============ chaotic mixer ============
class ChaoticBlock(nn.Module):
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


class ChaoticBase(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.block = ChaoticBlock(W, Wl, d, bl, bg)
        self.norm = nn.LayerNorm(d)

    def mix(self, x):
        h = self.embed(x) + self.pos
        return self.norm(h + self.block(h))


# V1: local readout only (no attention, no anchor)
class ModelV1(nn.Module):
    def __init__(self, base, V):
        super().__init__()
        self.base = base
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(),
                                     nn.Linear(D_MODEL, V))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


# V2: query-anchored dynamics — query identity re-injected each anchor round,
# context mixed around the anchor; readout is local (no attention anywhere)
class ModelV2(nn.Module):
    def __init__(self, base, V, rounds=ANCHOR_ROUNDS):
        super().__init__()
        self.base = base
        self.rounds = rounds
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(),
                                     nn.Linear(D_MODEL, V))

    def forward(self, x):
        B, W = x.shape
        h = self.base.embed(x) + self.base.pos
        h = self.base.norm(h + self.base.block(h))
        # anchor: the query token's state after first pass (its identity in context)
        anchor = h[:, -1:, :].detach().clone()
        for _ in range(self.rounds):
            h[:, -1:, :] = anchor          # re-inject the query anchor
            h = self.base.norm(h + self.base.block(h))
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


# ============ evaluate ============
def evaluate(model, W, n_samples=20000, beta=BETA):
    model.eval()
    nll_m, nll_p, acc_m, acc_p = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            logits = model(torch.tensor([ctx], dtype=torch.long))
            logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
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
    model.train()
    return {"n": n,
            "ppl_model": float(np.exp(np.mean(nll_m))),
            "ppl_prior": float(np.exp(np.mean(nll_p))),
            "ppl_gain": float(np.exp(np.mean(nll_m)) / np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_prior": acc_p / n}


def train(model, steps=TRAIN_STEPS, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    n = len(train_ids) - W - 1
    s = 0
    while s < steps:
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
        Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
        opt.zero_grad()
        loss = lossf(model(X), Y)
        loss.backward()
        opt.step()
        s += 1
        if s % 3000 == 0:
            print(f"  [{tag} {s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    return loss.item()


results = {"V": V, "W": W, "d": D_MODEL, "order": ORDER, "beta": BETA,
           "train_steps": TRAIN_STEPS, "anchor_rounds": ANCHOR_ROUNDS,
           "train_tokens": len(train_ids), "test_tokens": len(test_ids)}

for name, make in [("V1_local", lambda: ModelV1(ChaoticBase(V, W, W_LOCAL, D_MODEL,
                                                            BLOCKS_LOCAL, BLOCKS_GLOBAL), V)),
                   ("V2_anchor", lambda: ModelV2(ChaoticBase(V, W, W_LOCAL, D_MODEL,
                                                             BLOCKS_LOCAL, BLOCKS_GLOBAL), V))]:
    print(f"\n=== {name} ===")
    model = make()
    n_params = sum(p.numel() for p in model.parameters())
    ckpt = os.path.join(OUT, f"{name}.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded checkpoint ({n_params:,} params)")
    else:
        print(f"params: {n_params:,}")
        train(model, tag=name)
        torch.save(model.state_dict(), ckpt)
    r = evaluate(model, W)
    print(f"{name}: {r}")
    results[name] = {**r, "params": n_params}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
