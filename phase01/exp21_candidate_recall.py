"""exp21_candidate_recall.py — Phase 4, Experiment I (the GATE).

Q: Can chaotic routing find the connections that full attention considers
important? (Candidate Recall)

Setup (W=256, BPE-512, real code corpus):
  Teacher: AttnLM (exp13 checkpoint, single-head full attention over W).
  Mixer:   Hierarchical Chaotic Mixer (exp14 checkpoint mixer_1.pt) + random-init control.
  Scorings compared against the teacher's Top-K_attention:
    raw_qk   — q·k on UNMIXED states (control: does mixing help at all?)
    chaos_qk — q·h_j on MIXED chaotic states (the proposal)
    linear   — learned linear probe on mixed states (content-INDEPENDENT)
  Metrics per K in {4,8,16,32,64}: recall, precision, jaccard, top1-hit, hit-any,
  avg candidate distance; plus Spearman/Pearson (Exp VII: chaos vs attention).

Gate: Recall@16 > 50% interesting, >70% strong, >85% very serious.
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
OUT = os.path.join(HERE, "exp21_candidate_recall")
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
N_EVAL_WINDOWS = 1500
WINDOW_STRIDE = 32
KS = [4, 8, 16, 32, 64]


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


# ============ teacher: full attention LM (exp13 checkpoint) ============
class AttnLM(nn.Module):
    def __init__(self, V, W, d):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, W = x.shape
        h = self.embed(x) + self.pos
        qkv = self.qkv(h).reshape(B, W, 3, self.d).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = attn.softmax(-1)
        return self.norm(h + self.proj(attn @ v)), attn


teacher = AttnLM(V, W, D_MODEL)
ckpt_t = os.path.join(HERE, "exp13_two_selectors", "attnlm.pt")
if os.path.exists(ckpt_t):
    teacher.load_state_dict(torch.load(ckpt_t, weights_only=True))
    print("teacher loaded (exp13 attnlm.pt)")
else:
    raise SystemExit("no teacher checkpoint — run exp13 first")


# ============ chaotic mixer (exp14 checkpoint) ============
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


mixer = MultiLayerChaotic(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL, 1)
ckpt_m = os.path.join(HERE, "exp14_multilayer", "mixer_1.pt")
if os.path.exists(ckpt_m):
    mixer.load_state_dict(torch.load(ckpt_m, weights_only=True))
    print("mixer loaded (exp14 mixer_1.pt)")
else:
    raise SystemExit("no mixer checkpoint — run exp14 first")


# ============ collect teacher attention + states ============
teacher.eval()
mixer.eval()

starts = list(range(0, len(test_ids) - W - 1, WINDOW_STRIDE))[:N_EVAL_WINDOWS]
attn_rows = []      # teacher attention for the LAST query position, per window
h_raw_rows = []     # unmixed states (embed+pos), ALL positions
h_mix_rows = []     # mixed chaotic states, all positions
h_mix_query = []    # mixed state at query position
n_windows = 0

with torch.no_grad():
    for s in starts:
        ctx = test_ids[s:s + W]
        x = torch.tensor([ctx], dtype=torch.long)
        h_attn, attn = teacher(x)                      # attn (1, W, W)
        h_mix_full = mixer(x)                          # (1, W, d)
        h_raw_full = mixer.embed(x) + mixer.pos        # (1, W, d)
        q = W - 1
        attn_rows.append(attn[0, q, :].numpy())        # (W,)
        h_raw_rows.append(h_raw_full[0].numpy())       # (W, d)
        h_mix_rows.append(h_mix_full[0].numpy())
        h_mix_query.append(h_mix_full[0, q, :].numpy())
        n_windows += 1

attn_rows = np.stack(attn_rows)          # (N, W)
h_raw_rows = np.stack(h_raw_rows)        # (N, W, d)
h_mix_rows = np.stack(h_mix_rows)        # (N, W, d)
h_mix_query = np.stack(h_mix_query)      # (N, d)
print(f"collected {n_windows} windows")

# ============ scorings ============
# chaos_qk: query state · candidate state (mixed)
chaos_scores = (h_mix_query[:, None, :] * h_mix_rows).sum(-1)     # (N, W)
# raw_qk: unmixed query · unmixed candidate
raw_scores = (h_mix_query[:, None, :] * 0 + 1)  # placeholder
raw_query = h_raw_rows[:, -1, :]                                   # (N, d)
raw_scores = (raw_query[:, None, :] * h_raw_rows).sum(-1)          # (N, W)

# linear probe: fit w on TRAIN windows (content-independent selector)
print("fitting linear probe on train windows...")
tr_starts = list(range(0, len(train_ids) - W - 1, 64))[:800]
Xs, ys = [], []
with torch.no_grad():
    for s in tr_starts:
        x = torch.tensor([train_ids[s:s + W]], dtype=torch.long)
        _, attn = teacher(x)
        h_full = mixer(x)
        q = W - 1
        Xs.append(h_full[0].numpy())                   # (W, d)
        ys.append(attn[0, q, :].numpy())               # (W,)
Xs = np.concatenate(Xs)                                 # (M, d)
ys = np.concatenate(ys)                                 # (M,)
lam = 1e-3
w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(D_MODEL), Xs.T @ ys)   # (d,)
linear_scores = h_mix_rows @ w                          # (N, W)
print(f"linear probe fit (R2={1 - np.var(ys - Xs @ w)/np.var(ys):.3f})")


def topk_indices(scores, k):
    return np.argsort(-scores, axis=1)[:, :k]


def metrics(attn_rows, scores, ks=KS):
    out = {}
    for k in ks:
        top_attn = topk_indices(attn_rows, k)     # (N, k)
        top_chaos = topk_indices(scores, k)
        n = len(attn_rows)
        inter = np.array([len(set(top_attn[i]) & set(top_chaos[i])) for i in range(n)])
        recall = inter.mean() / k
        prec = inter.mean() / k                    # symmetric for top-k sets
        jac = np.array([len(set(top_attn[i]) & set(top_chaos[i])) /
                        len(set(top_attn[i]) | set(top_chaos[i])) for i in range(n)])
        top1_hit = np.array([top_chaos[i, 0] in set(top_attn[i]) for i in range(n)])
        any_hit = (inter > 0).mean()
        # average positional distance of chaotic candidates from query
        dist = np.array([np.abs(top_chaos[i] - (W - 1)).mean() for i in range(n)])
        out[f"K{k}"] = {"recall": float(recall), "precision": float(prec),
                        "jaccard": float(jac.mean()), "top1_hit": float(top1_hit.mean()),
                        "any_hit": float(any_hit),
                        "avg_cand_dist": float(dist.mean()),
                        "attention_avg_dist": float(np.abs(
                            np.array([top_attn[i] - (W - 1) for i in range(n)])).mean())}
    return out


def spearman(a, b):
    """Spearman = Pearson on ranks. numpy-only."""
    def rankdata(x):
        sorter = np.argsort(x)
        ranks = np.empty(len(x), dtype=np.float64)
        ranks[sorter] = np.arange(len(x))
        return ranks
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    pearson = (ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum() + 1e-12)
    # Pearson on raw values
    aa, bb = a - a.mean(), b - b.mean()
    pe = (aa * bb).sum() / np.sqrt((aa ** 2).sum() * (bb ** 2).sum() + 1e-12)
    return float(pearson), float(pe)


print("\n=== Candidate Recall ===")
results = {"n_windows": n_windows, "W": W, "stride": WINDOW_STRIDE,
           "K_values": KS}
for name, scores in [("chaos_qk", chaos_scores), ("raw_qk", raw_scores),
                     ("linear_probe", linear_scores)]:
    m = metrics(attn_rows, scores)
    print(f"\n--- {name} ---")
    print("K | recall | precision | jaccard | top1_hit | any_hit | cand_dist")
    for k in KS:
        r = m[f"K{k}"]
        print(f"{k:>3} | {r['recall']:.3f} | {r['precision']:.3f} | {r['jaccard']:.3f} "
              f"| {r['top1_hit']:.3f} | {r['any_hit']:.3f} | {r['avg_cand_dist']:.0f}")
    results[name] = m

# ============ Exp VII: correlation chaos vs attention ============
print("\n=== Chaos vs Attention correlation (Exp VII) ===")
flat_chaos = chaos_scores.flatten()
flat_attn = attn_rows.flatten()
# sample for speed
idx = np.random.RandomState(0).choice(len(flat_chaos), 200_000, replace=False)
sp, pe = spearman(flat_chaos[idx], flat_attn[idx])
print(f"Spearman(chaos_qk, attention) = {sp:.4f}")
print(f"Pearson(chaos_qk, attention)  = {pe:.4f}")
results["correlation"] = {"spearman": sp, "pearson": pe}

# top-1 attention hit rate (spec metric)
top1_attn = attn_rows.argmax(-1)
chaos_top1 = chaos_scores.argmax(-1)
print(f"chaos top-1 == attention top-1: {(chaos_top1 == top1_attn).mean():.3f}")
results["chaos_top1_eq_attn_top1"] = float((chaos_top1 == top1_attn).mean())

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
