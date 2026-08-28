"""exp22_memory_scaling.py — Phase 5, exp22: the Memory-Scaling test.

Tests the β-Architecture hypothesis: does accuracy grow with the MEMORY
channel (corpus table) at FIXED (tiny) compute channel (mixer)?

Setup: reuse exp18 V1 (tiny chaotic mixer + local readout, 6K steps) —
the compute channel is FIXED. Vary ONLY the β-table:
  order-1 (unigram), order-2 capped (3K/10K/28K contexts), order-3 (full).
Measure PPL / top-1 / table memory. If PPL improves with table size while
the mixer stays fixed → the memory-scaling axis exists (new architecture
claim). If it saturates instantly → memory does not scale.

Also reports the table-alone PPL (memory channel standalone, no mixer).
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
OUT = os.path.join(HERE, "exp22_memory_scaling")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
MAX_TRAIN_BYTES = 2_000_000
BETA = 0.3
N_EVAL = 20000


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


# ============ prior builders (order-1/2/3, capped) ============
def build_prior(ids, order, cap=None):
    """Return {ctx_tuple: {token: count}}, context length = order."""
    prior = defaultdict(lambda: defaultdict(int))
    for i in range(order, len(ids)):
        prior[tuple(ids[i - order:i])][ids[i]] += 1
    prior = {k: dict(v) for k, v in prior.items()}
    if cap is not None:
        # keep the `cap` most frequent contexts
        ranked = sorted(prior.items(), key=lambda kv: -sum(kv[1].values()))[:cap]
        prior = dict(ranked)
    return prior


def make_logp_table(prior, order, V):
    """(context bytes) -> np array of log probs over V."""
    table = {}
    for ctx, cnts in prior.items():
        tot = sum(cnts.values())
        out = np.full(V, -1e9, dtype=np.float64)
        for t, c in cnts.items():
            out[t] = np.log(c / tot)
        table[ctx] = out
    return table


# ============ model (exp18 V1: tiny mixer + local readout) ============
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
ckpt = os.path.join(HERE, "exp18_no_attention", "V1_local.pt")
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print("V1 model loaded (exp18, fixed tiny mixer + local readout)")
else:
    raise SystemExit("no V1 checkpoint — run exp18 first")
model.eval()

n_params_mixer = sum(p.numel() for p in model.base.parameters())
n_params_total = sum(p.numel() for p in model.parameters())
print(f"compute-channel params: mixer={n_params_mixer:,}, total={n_params_total:,}")


# ============ evaluation with a given prior ============
def evaluate_with_prior(logp_table, order, n_samples=N_EVAL, beta=BETA):
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
            # prior lookup
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
    return {"n": n,
            "ppl_model": float(np.exp(np.mean(nll_m))),
            "ppl_prior": float(np.exp(np.mean(nll_p))),
            "acc_model": acc_m / n, "acc_prior": acc_p / n}


def prior_alone_ppl(logp_table, order, n_samples=N_EVAL):
    """PPL of the table alone (memory channel standalone)."""
    nll = []
    for i in range(0, len(test_ids) - W - 1, 32):
        ctx = test_ids[i:i + W]
        y = test_ids[i + W]
        cp = None
        for back in range(order, 0, -1):
            t = logp_table.get(tuple(ctx[-back:]))
            if t is not None:
                cp = t
                break
        if cp is not None:
            nll.append(-cp[y])
        else:
            nll.append(np.log(V))  # uniform backoff
        if len(nll) >= n_samples:
            break
    return float(np.exp(np.mean(nll)))


results = {"V": V, "W": W, "beta": BETA, "n_eval": N_EVAL,
           "mixer_params": n_params_mixer, "total_params": n_params_total}

# memory channel variants: increasing table size
variants = [
    ("order1_unigram", 1, None),
    ("order2_cap3k", 2, 3000),
    ("order2_cap10k", 2, 10000),
    ("order2_full", 2, None),
    ("order3_full", 3, None),
]

print("\n=== Memory-Scaling (fixed tiny mixer) ===")
print("variant | table_ctx | table_mem_MB | PPL_model | PPL+β | top1+β | table_alone_PPL")
for name, order, cap in variants:
    prior = build_prior(train_ids, order, cap)
    logp_table = make_logp_table(prior, order, V)
    n_ctx = len(prior)
    n_assoc = sum(len(v) for v in prior.values())
    mem_mb = (n_ctx * order * 4 + n_assoc * 2 * 4) / 1e6
    r = evaluate_with_prior(logp_table, order)
    ta = prior_alone_ppl(logp_table, order)
    print(f"{name:16s} | {n_ctx:>8,} | {mem_mb:>8.2f} | {r['ppl_model']:>7.2f} "
          f"| {r['ppl_prior']:>6.2f} | {r['acc_prior']:>6.3f} | {ta:>7.2f}")
    results[name] = {**r, "table_contexts": n_ctx, "table_assoc": n_assoc,
                     "table_mem_MB": mem_mb, "table_alone_ppl": ta}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
