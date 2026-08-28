"""exp37_generation_v07.py — text generation with Architecture v0.7.

The final architecture: standalone chaotic mixer + sparse MLE memory +
confidence-gated β(c_h). This is the generation demo — does it produce
coherent text, not just low PPL?

Generation: autoregressive sampling (temp 0.8, top-p 0.9) from
P(w) = (1-β(c_h))·p_mix(w) + β(c_h)·p_sparse(w).
β(c_h) = βmax·c_h/(c_h + k), k=0.5, βmax=1.0 (tuned on both domains).

Shows samples on CODE and NL (WikiText-2).
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
OUT = os.path.join(HERE, "exp37_generation_v07")
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
K_BETA = 0.5
BETA_MAX = 1.0
GEN_LEN = 140
TEMP = 0.8
TOPP = 0.9


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()[:limit] if limit else f.read()


def make_bpe(text, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


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
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, W, D_MODEL) * 0.02)
        self.block = ChaoticBlock()
        self.norm = nn.LayerNorm(D_MODEL)

    def mix(self, x):
        return self.norm(self.embed(x) + self.pos + self.block(self.embed(x) + self.pos))


class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ChaoticBase()
        self.readout = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, VOCAB_SIZE))

    def forward(self, x):
        h = self.base.mix(x)
        gvec = h.mean(dim=1)
        return self.readout(torch.cat([h[:, -1, :], gvec], dim=-1))


def build_domain(name, train_path, test_path, ckpt_path, max_chars):
    """Returns dict with tok, ids, ctx_counts, model."""
    print(f"--- {name} ---")
    train_text = load_chars(train_path, max_chars)
    test_text = load_chars(test_path)
    tok = make_bpe(train_text)
    train_ids = tok.encode(train_text).ids
    test_ids = tok.encode(test_text).ids
    ctx_counts = defaultdict(dict)
    for i in range(ORDER, len(train_ids)):
        ctx = tuple(train_ids[i - ORDER + 1:i])
        w = train_ids[i]
        d_ = ctx_counts[ctx]
        d_[w] = d_.get(w, 0) + 1
    model = ModelV1()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    print(f"  V={tok.get_vocab_size()} train={len(train_ids):,} contexts={len(ctx_counts):,}")
    return {"tok": tok, "train_ids": train_ids, "test_ids": test_ids,
            "ctx_counts": ctx_counts, "model": model, "V": tok.get_vocab_size()}


def sparse_logp_mem(ctx_counts, ctx, V):
    """Full sparse logp array (observed continuations get MLE, rest -1e9)."""
    logp = np.full(V, -1e9, dtype=np.float64)
    e = ctx_counts.get(ctx)
    if e:
        tot = sum(e.values())
        for w, c in e.items():
            logp[w] = np.log(c / tot)
    return logp


def generate(dom, seed_text, length=GEN_LEN):
    tok = dom["tok"]
    model = dom["model"]
    ctx_counts = dom["ctx_counts"]
    V = dom["V"]
    ids = tok.encode(seed_text).ids
    out = seed_text
    betas = []
    for _ in range(length):
        ctx = ids[-W:] if len(ids) >= W else ids
        fill = ctx[0] if ctx else 0
        full = [fill] * (W - len(ctx)) + ctx
        x = torch.tensor([full], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        lp_mix = torch.log_softmax(logits[0], -1).double().numpy()
        pm = np.exp(lp_mix)
        # memory
        mem_ctx = tuple(ids[-ORDER + 1:]) if len(ids) >= ORDER - 1 else ()
        e = ctx_counts.get(mem_ctx)
        if e:
            c_h = sum(e.values())
            beta = BETA_MAX * c_h / (c_h + K_BETA)
            pmem = np.zeros(V)
            for w, c in e.items():
                pmem[w] = c / c_h
            pf = (1 - beta) * pm + beta * pmem
        else:
            beta = 0.0
            pf = pm
        betas.append(beta)
        # sample (temp, top-p)
        p = np.maximum(pf, 1e-12)
        p = p ** (1 / TEMP)
        p /= p.sum()
        idx = np.argsort(-p)
        cum = np.cumsum(p[idx])
        keep = idx[cum <= TOPP]
        if len(keep) == 0:
            keep = idx[:1]
        mask = np.zeros(V, dtype=bool)
        mask[keep] = True
        p[~mask] = 0
        p /= p.sum()
        nxt = int(np.random.choice(V, p=p))
        ids.append(nxt)
        out += tok.decode([nxt])
    return out, float(np.mean(betas))


# ---- CODE ----
code = build_domain("CODE", os.path.join(HERE, "corpus_train.txt"),
                    os.path.join(HERE, "corpus_test.txt"),
                    os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                    2_000_000)

# ---- NL ----
nl = build_domain("NL", os.path.join(HERE, "nl_corpus", "nl_corpus_train.txt"),
                  os.path.join(HERE, "nl_corpus", "nl_corpus_test.txt"),
                  os.path.join(HERE, "exp23_nl_beta", "nl_mixer.pt"),
                  4_000_000)

results = {}

code_seeds = [
    ("code_fn", "def fibonacci(n):"),
    ("code_api", "app.get('/users', async (req, res) => {"),
    ("code_import", "import React, { useState } from 'react';"),
]
print("\n========== CODE GENERATION (v0.7) ==========")
results["code"] = {}
for sname, seed in code_seeds:
    torch.manual_seed(0)
    np.random.seed(0)
    gen, avg_beta = generate(code, seed)
    print(f"\n--- {sname} (avg β {avg_beta:.2f}) ---\n{gen}")
    results["code"][sname] = {"seed": seed, "text": gen, "avg_beta": avg_beta}

nl_seeds = [
    ("nl_art", "The history of artificial intelligence began"),
    ("nl_energy", "Solar energy is one of the most promising"),
    ("nl_novel", "The old lighthouse stood on the cliff"),
]
print("\n========== NL GENERATION (v0.7) ==========")
results["nl"] = {}
for sname, seed in nl_seeds:
    torch.manual_seed(0)
    np.random.seed(0)
    gen, avg_beta = generate(nl, seed)
    print(f"\n--- {sname} (avg β {avg_beta:.2f}) ---\n{gen}")
    results["nl"][sname] = {"seed": seed, "text": gen, "avg_beta": avg_beta}

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=2)
print("\nsaved", OUT)
