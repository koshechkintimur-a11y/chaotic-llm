"""exp23_vis_data.py — extract REAL prediction data for the HTML animation.

Loads the trained tiny mixer (exp18 V1) + order-3 β-table, runs real code
contexts through all three channels (mixer-only, table-only, β-fused) and
dumps full logits + top predictions + scaling curve to vis_data.json.
The HTML then renders the pipeline, the β-slider and the memory-scaling
chart from this data.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaos_lib import permute_indices

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp22_memory_scaling")
os.makedirs(OUT, exist_ok=True)

VOCAB_SIZE = 512
W = 256
W_LOCAL = 64
BLOCKS_LOCAL = 8
BLOCKS_GLOBAL = 4
D_MODEL = 64
MAX_TRAIN_BYTES = 2_000_000
N_CONTEXTS = 4


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


tok = make_bpe(train_text)
V = tok.get_vocab_size()
train_ids = tok.encode(train_text).ids
test_ids = tok.encode(test_text).ids


# ============ order-3 prior ============
prior = defaultdict(lambda: defaultdict(int))
for i in range(3, len(train_ids)):
    prior[tuple(train_ids[i - 3:i])][train_ids[i]] += 1
prior = {k: dict(v) for k, v in prior.items()}

logp_table = {}
for ctx, cnts in prior.items():
    tot = sum(cnts.values())
    out = np.full(V, -1e9, dtype=np.float64)
    for t, c in cnts.items():
        out[t] = np.log(c / tot)
    logp_table[ctx] = out


def table_logp(ids, i):
    """order-3 then 2 then 1 backoff table logp for predicting ids[i]."""
    for back in range(3, 0, -1):
        t = logp_table.get(tuple(ids[i - back:i]))
        if t is not None:
            return t, back
    return None, 0


# ============ model ============
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
model.load_state_dict(torch.load(os.path.join(HERE, "exp18_no_attention", "V1_local.pt"),
                                 weights_only=True))
model.eval()

# ============ pick real contexts ============
# find contexts where the table has an order-3 match and the answer is non-trivial
starts = list(range(200, len(test_ids) - W - 1, 137))
picked = []
for s in starts:
    if len(picked) >= N_CONTEXTS:
        break
    i = s + W  # prediction position
    if table_logp(test_ids, i)[0] is not None:
        picked.append(s)


def topn(logp, n=8):
    order = np.argsort(-logp)[:n]
    return [(int(t), float(logp[t])) for t in order]


# decode tokens to text (with readable separators)
def tokens_text(ids):
    raw = tok.decode(ids)
    return raw


vis = {"vocab_size": V, "contexts": [], "scaling": None}

with torch.no_grad():
    for s in picked:
        ctx_ids = test_ids[s:s + W]
        y = test_ids[s + W]
        logits = model(torch.tensor([ctx_ids], dtype=torch.long))
        logp_mixer = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
        lp_table, back = table_logp(test_ids, s + W)
        if lp_table is None:
            continue
        # fused at several beta
        def fuse(beta):
            return np.logaddexp(np.log1p(-beta) + logp_mixer, np.log(beta) + lp_table)
        ctx_text = tokens_text(ctx_ids)
        tail = tokens_text(ctx_ids[-8:])  # the recent 8 tokens (table context lives here)
        entry = {
            "context_text": ctx_text[-220:],
            "tail_text": tail,
            "correct": y,
            "correct_text": tok.decode([y]),
            "table_backoff": back,
            "logp_mixer": logp_mixer.tolist(),
            "logp_table": lp_table.tolist(),
            "fused_0.3": fuse(0.3).tolist(),
            "fused_0.97": fuse(0.97).tolist(),
            "top_mixer": topn(logp_mixer),
            "top_table": topn(lp_table),
            "top_fused": topn(fuse(0.3)),
        }
        vis["contexts"].append(entry)

# coverage stats
n_cov = 0
n_all = 0
for i in range(3000, min(len(test_ids) - 1, 100000)):
    n_all += 1
    if table_logp(test_ids, i)[0] is not None:
        n_cov += 1
vis["coverage"] = {"test_positions": n_all, "order3_matches": n_cov,
                   "order3_coverage": n_cov / n_all}

# memory scaling curve (from exp22 results)
vis["scaling"] = [
    {"mem": 0.23, "ppl": 28.34, "acc": 0.232, "label": "order-1"},
    {"mem": 0.45, "ppl": 20.14, "acc": 0.324, "label": "order-2 3K"},
    {"mem": 0.86, "ppl": 15.39, "acc": 0.384, "label": "order-2 10K"},
    {"mem": 1.27, "ppl": 13.79, "acc": 0.407, "label": "order-2 full"},
    {"mem": 3.54, "ppl": 10.35, "acc": 0.524, "label": "order-3 full"},
]
vis["best"] = {"ppl": 10.35, "acc": 0.524, "mixer_alone_ppl": 32.83}
vis["references"] = {"transformer_ppl": 11.9, "transformer_acc": 0.424}

with open(os.path.join(OUT, "vis_data.json"), "w", encoding="utf-8") as f:
    json.dump(vis, f, ensure_ascii=False)
print("saved", os.path.join(OUT, "vis_data.json"))
print("contexts:", len(vis["contexts"]))
for c in vis["contexts"]:
    print("  correct:", c["correct_text"], "| tail:", repr(c["tail_text"][-40:]))
print("coverage order-3:", vis["coverage"]["order3_coverage"])
