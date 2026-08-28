"""exp23_nl_beta.py — Phase 5, exp23: the β-law on NATURAL LANGUAGE.

Q: does the β-Architecture's memory channel help on natural language the way
it does on code (β=0.972)? If β stays high (>0.5) — memory is universally
useful (strong architecture claim). If β drops — memory is code-specific.

Setup (mirror of the code experiment exp22):
  Corpus: WikiText-2 (10.9M train chars).
  Tokenizer: BPE-512 trained on NL.
  Compute channel: SAME tiny hierarchical chaotic mixer + local readout
    (trained fresh on NL, 6000 steps — same budget as exp18).
  Memory channel: order-1/2/3 n-gram tables from NL train.
  Eval: mixer alone, fused at β sweep, best β; coverage stats.
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
OUT = os.path.join(HERE, "exp23_nl_beta")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
BATCH = 64
MAX_TRAIN_CHARS = 4_000_000
TRAIN_STEPS = 6000
N_EVAL = 20000


def load_chars(path, limit=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[:limit] if limit else text


train_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_train.txt"),
                        MAX_TRAIN_CHARS)
test_text = load_chars(os.path.join(HERE, "nl_corpus", "nl_corpus_test.txt"))
print(f"NL train chars: {len(train_text):,}, test chars: {len(test_text):,}")


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


# ============ prior builders ============
def build_prior(ids, order, cap=None):
    prior = defaultdict(lambda: defaultdict(int))
    for i in range(order, len(ids)):
        prior[tuple(ids[i - order:i])][ids[i]] += 1
    prior = {k: dict(v) for k, v in prior.items()}
    if cap is not None:
        ranked = sorted(prior.items(), key=lambda kv: -sum(kv[1].values()))[:cap]
        prior = dict(ranked)
    return prior


def make_logp_table(prior, V):
    table = {}
    for ctx, cnts in prior.items():
        tot = sum(cnts.values())
        out = np.full(V, -1e9, dtype=np.float64)
        for t, c in cnts.items():
            out[t] = np.log(c / tot)
        table[ctx] = out
    return table


# ============ model (tiny chaotic mixer + local readout) ============
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


model = ModelV1(ChaoticBase(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL), V)
ckpt = os.path.join(OUT, "nl_mixer.pt")
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print("loaded NL mixer checkpoint")
else:
    print(f"\ntraining tiny mixer on NL ({sum(p.numel() for p in model.parameters()):,} params)")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    n = len(train_ids) - W - 1
    for s in range(TRAIN_STEPS):
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
        Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
        opt.zero_grad()
        loss = lossf(model(X), Y)
        loss.backward()
        opt.step()
        if s % 3000 == 0:
            print(f"  [mixer {s:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), ckpt)
    print("mixer trained + saved")


# ============ evaluate ============
def evaluate(logp_table, order, beta, n_samples=N_EVAL):
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
            cp = None
            for back in range(order, 0, -1):
                t = logp_table.get(tuple(ctx[-back:]))
                if t is not None:
                    cp = t
                    break
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
            "ppl_fused": float(np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_fused": acc_p / n}


results = {"corpus": "wikitext-2", "V": V, "W": W, "train_steps": TRAIN_STEPS,
           "train_tokens": len(train_ids), "test_tokens": len(test_ids),
           "mixer_params": sum(p.numel() for p in model.parameters())}

# order-3 table + coverage
prior3 = build_prior(train_ids, 3)
logp3 = make_logp_table(prior3, V)
print(f"order-3 table: {len(prior3):,} contexts")

# coverage on test
cov = 0
n_all = 0
for i in range(3000, min(len(test_ids) - 1, 100000)):
    n_all += 1
    if tuple(test_ids[i - 3:i]) in logp3:
        cov += 1
print(f"order-3 coverage on NL test: {cov/n_all:.3f}")
results["order3_coverage"] = cov / n_all

# mixer alone
r0 = evaluate(logp3, 3, 0.0)
print(f"\nmixer alone: PPL {r0['ppl_model']:.2f}, top-1 {r0['acc_model']*100:.1f}%")
results["mixer_alone"] = {"ppl": r0["ppl_model"], "acc": r0["acc_model"]}

# β sweep
print("\n=== β sweep (order-3, NL) ===")
print("beta | PPL_fused | top-1")
best_beta, best_ppl = 0.0, r0["ppl_model"]
sweep = {}
for beta in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99]:
    r = evaluate(logp3, 3, beta)
    print(f"{beta:.2f} | {r['ppl_fused']:.2f} | {r['acc_fused']*100:.1f}%")
    sweep[str(beta)] = {"ppl": r["ppl_fused"], "acc": r["acc_fused"]}
    if r["ppl_fused"] < best_ppl:
        best_ppl, best_beta = r["ppl_fused"], beta
results["beta_sweep"] = sweep
results["best_beta_nl"] = best_beta
results["best_ppl_nl"] = best_ppl
print(f"\n→ best β on natural language: {best_beta} (PPL {best_ppl:.2f})")
print(f"  reference: best β on CODE = 0.972 (PPL 10.35)")

# memory scaling on NL (order-1/2/3 at fixed best-ish beta 0.5)
print("\n=== memory-scaling on NL (β=0.5) ===")
for order in [1, 2, 3]:
    pr = build_prior(train_ids, order)
    lp = make_logp_table(pr, V)
    r = evaluate(lp, order, 0.5)
    mem = (len(pr) * order * 4 + sum(len(v) for v in pr.values()) * 2 * 4) / 1e6
    print(f"order-{order}: ctx={len(pr):>8,} mem={mem:>6.2f}MB PPL_fused={r['ppl_fused']:.2f} "
          f"top1={r['acc_fused']*100:.1f}%")
    results[f"nl_order{order}"] = {"ctx": len(pr), "mem_MB": mem,
                                   "ppl_fused": r["ppl_fused"],
                                   "acc_fused": r["acc_fused"]}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
