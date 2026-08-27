"""exp17_knn_lm.py — Step 5d: kNN-LM vs beta-prior at EQUAL memory budget.

The honest engineering question from morin-filter: is the aggregated n-gram
prior (beta-prior) better than kNN-LM retrieval at the same memory budget?

Setup (same base model L=1 chaotic mixer + attention readout from exp14):
  1. model alone
  2. model + beta-prior (order-2 BPE table, aggregated counts)
  3. model + kNN-LM with datastore capped to the SAME memory (bytes) as the table
  4. model + large kNN-LM datastore (no cap) as reference

kNN keys = random 8-dim projection of the model's last-position hidden state.
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
OUT = os.path.join(HERE, "exp17_knn_lm")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

VOCAB_SIZE = 512
W_LOCAL = 64
W = 256
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
MAX_TRAIN_BYTES = 2_000_000
ORDER = 2
BETA = 0.3
LAMBDA = 0.3
K_NEIGH = 16
KEY_DIM = 8
TEST_N = 20000


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
n_ctx = len(prior)
n_assoc = sum(len(v) for v in prior.values())
print(f"prior contexts: {n_ctx:,}, (ctx->next) associations: {n_assoc:,}")

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


# ============ model (reuse exp14 L1) ============
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


class MultiLayerChaotic(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg, L=1):
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


mixer = MultiLayerChaotic(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL, 1)
readout = AttnReadout(V, W, D_MODEL)
ckpt_m = os.path.join(HERE, "exp14_multilayer", "mixer_1.pt")
ckpt_r = os.path.join(HERE, "exp14_multilayer", "readout_1.pt")
if os.path.exists(ckpt_m):
    mixer.load_state_dict(torch.load(ckpt_m, weights_only=True))
    readout.load_state_dict(torch.load(ckpt_r, weights_only=True))
    print("loaded L=1 from exp14")
else:
    raise SystemExit("exp14 L1 checkpoint not found — run exp14 first")


# ============ datastore: hidden states over training ============
print("Extracting hidden states over training set...")
mixer.eval()
readout.eval()
n_train = len(train_ids) - W - 1
stride_full = 1
n_full = n_train // stride_full
full_states = np.empty((n_full, D_MODEL), dtype=np.float32)
full_next = np.empty(n_full, dtype=np.int64)
with torch.no_grad():
    idx = 0
    for start in range(0, n_train, 256):  # batch of windows
        ends = min(start + 256, n_train)
        X = np.stack([train_ids[i:i + W] for i in range(start, ends)])
        h = mixer(torch.tensor(X, dtype=torch.long))
        full_states[idx:idx + h.shape[0]] = h[:, -1, :].numpy()
        full_next[idx:idx + h.shape[0]] = [train_ids[i + W] for i in range(start, ends)]
        idx += h.shape[0]
full_states = full_states[:idx]
full_next = full_next[:idx]
print(f"datastore full: {idx:,} entries")

# random projection 64 -> 8 dims (fixed seed, orthogonal-ish by normalization)
rng = np.random.default_rng(42)
PROJ = rng.standard_normal((D_MODEL, KEY_DIM), dtype=np.float32)
PROJ /= np.linalg.norm(PROJ, axis=0, keepdims=True)


def project(h):
    return (h @ PROJ).astype(np.float32)


# memory accounting
table_bytes = n_ctx * 2 * 4 + n_assoc * 3 * 4   # ctx keys + counts
entry_bytes = KEY_DIM * 4 + 4                     # key + next token
K_cap = int(table_bytes / entry_bytes)
print(f"beta-table memory ~{table_bytes/1e6:.2f} MB; kNN entry {entry_bytes} B; "
      f"equal-memory K = {K_cap:,}")


def knn_selector(store_keys, store_next, test_states, k=K_NEIGH, chunk=64):
    """Return list of np arrays: p_knn over vocab per test query."""
    store_keys = project(store_keys)
    store_norm = np.linalg.norm(store_keys, axis=1, keepdims=True) + 1e-8
    store_keys = store_keys / store_norm
    test_keys = project(test_states)
    test_keys = test_keys / (np.linalg.norm(test_keys, axis=1, keepdims=True) + 1e-8)
    out = []
    for c0 in range(0, len(test_keys), chunk):
        c1 = min(c0 + chunk, len(test_keys))
        sim = test_keys[c0:c1] @ store_keys.T          # (chunk, K)
        top = np.argpartition(-sim, k, axis=1)[:, :k]  # top-k indices
        top_sim = np.take_along_axis(sim, top, axis=1)
        w = np.exp(top_sim)                              # distance -> weight
        w = w / w.sum(axis=1, keepdims=True)
        toks = store_next[top]
        for row in range(len(top)):
            pk = np.zeros(V, dtype=np.float64)
            for j in range(k):
                pk[toks[row, j]] += w[row, j]
            out.append(pk)
    return out


def evaluate_model_only(n_samples=TEST_N):
    nll, acc = [], 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            h = mixer(torch.tensor(ctx, dtype=torch.long).unsqueeze(0))
            logits = readout(h)
            logp = torch.log_softmax(logits[0], -1).numpy()
            nll.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc += 1
            if len(nll) >= n_samples:
                break
    n = len(nll)
    return {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n}


def evaluate_with_prior(n_samples=TEST_N, beta=BETA):
    nll, acc = [], 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            h = mixer(torch.tensor(ctx, dtype=torch.long).unsqueeze(0))
            logits = readout(h)
            logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
            cp = prior_logp(ctx)
            if cp is not None:
                logp = np.logaddexp(np.log1p(-beta) + logp, np.log(beta) + cp)
            nll.append(-logp[y])
            if int(np.argmax(logp)) == y:
                acc += 1
            if len(nll) >= n_samples:
                break
    n = len(nll)
    return {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n}


def evaluate_with_knn(store_keys, store_next, n_samples=TEST_N, lam=LAMBDA):
    nll, acc = [], 0
    states = []
    ys = []
    ctxs = []
    window_starts = list(range(0, min(len(test_ids) - W - 1, n_samples * 32 + 1), 32))
    with torch.no_grad():
        for c0 in range(0, len(window_starts), 32):
            batch = window_starts[c0:c0 + 32]
            X = np.stack([test_ids[s:s + W] for s in batch])
            h = mixer(torch.tensor(X, dtype=torch.long))
            states.extend(h[:, -1, :].numpy())
            for s in batch:
                ys.append(test_ids[s + W])
                ctxs.append(test_ids[s:s + W])
    states = np.stack(states)
    ys = np.array(ys)
    print(f"  kNN retrieval over {len(store_keys):,} keys...")
    t0 = time.time()
    pk_list = knn_selector(store_keys, store_next, states)
    print(f"  retrieval done in {time.time()-t0:.0f}s")
    # batched model evaluation
    model_logp = []
    with torch.no_grad():
        for c0 in range(0, len(ys), 32):
            X = torch.tensor(np.stack(ctxs[c0:c0 + 32]), dtype=torch.long)
            h = mixer(X)
            logits = readout(h)
            model_logp.append(torch.log_softmax(logits, -1).numpy().astype(np.float64))
    model_logp = np.concatenate(model_logp)
    for j in range(len(ys)):
        logp = model_logp[j]
        pk = pk_list[j]
        if pk.sum() > 0:
            pk = pk / pk.sum()
            logp = np.logaddexp(np.log1p(-lam) + logp, np.log(lam) + np.log(pk + 1e-12))
        nll.append(-logp[ys[j]])
        if int(np.argmax(logp)) == ys[j]:
            acc += 1
    n = len(nll)
    return {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n,
            "K": len(store_keys)}


results = {"V": V, "W": W, "order": ORDER, "beta": BETA, "lambda": LAMBDA,
           "k_neigh": K_NEIGH, "key_dim": KEY_DIM,
           "table_memory_bytes": table_bytes, "knn_entry_bytes": entry_bytes,
           "K_equal_memory": K_cap, "datastore_full": idx}

# 1) model alone
print("\n=== 1) model alone ===")
r1 = evaluate_model_only()
print(r1)
results["model_alone"] = r1

# 2) model + beta-prior
print("\n=== 2) model + beta-prior ===")
r2 = evaluate_with_prior()
print(r2)
results["beta_prior"] = r2

# 3) model + kNN at equal memory (random subsample of full datastore)
print("\n=== 3) model + kNN equal-memory ===")
rng2 = np.random.default_rng(7)
perm = rng2.permutation(idx)[:K_cap]
r3 = evaluate_with_knn(full_states[perm], full_next[perm])
print(r3)
results["knn_equal_memory"] = r3

# 4) model + large kNN (reference, no cap)
print("\n=== 4) model + large kNN (reference) ===")
r4 = evaluate_with_knn(full_states, full_next)
print(r4)
results["knn_large"] = r4

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
