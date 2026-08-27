"""exp12_full_architecture.py — Phase 0.1, Experiment 12 (Step 3).

FULL ARCHITECTURE v0.2:  Hierarchical ChaoticLM + beta corpus-prior.

The synthesis assembled:
  - cheap reversible hierarchical chaotic mixing (proposer)
  - + beta corpus-prior filter (selector, from morin-filter)
  - vs the same beta-prior on a transformer baseline.

Reuses the exp11 HierChaoticLM checkpoint (must verify the tokenizer
rebuilt deterministically — sanity check PPL ~25.5 on a small sample).
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
OUT = os.path.join(HERE, "exp12_full_architecture")
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
TRAIN_STEPS = 15500


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


# ============ beta corpus-prior (BPE n-grams, TRAIN only) ============
print("Building beta prior...")
prior = defaultdict(lambda: defaultdict(int))
for i in range(ORDER, len(train_ids)):
    ctx = tuple(train_ids[i - ORDER:i])
    prior[ctx][train_ids[i]] += 1
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


# ============ models ============
class HierChaoticLM(nn.Module):
    def __init__(self, V, W, Wl, d, bl, bg):
        super().__init__()
        self.W, self.Wl, self.Nw = W, Wl, W // Wl
        self.d = d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.gates_l = nn.Parameter(torch.zeros(bl))
        self.gates_g = nn.Parameter(torch.zeros(bg))
        self._sig_l = {t: torch.as_tensor(permute_indices(Wl, t), dtype=torch.long)
                       for t in range(1, bl + 1)}
        self._sig_g = {t: torch.as_tensor(permute_indices(self.Nw, t), dtype=torch.long)
                       for t in range(1, bg + 1)}
        self.readout = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, V))

    def _chaotic(self, h, sigmas, gates):
        B, N, d = h.shape
        for t in range(1, len(gates) + 1):
            h = h[:, sigmas[t].to(h.device), :]
            g = torch.sigmoid(gates[t - 1])
            even = h[:, 0::2, :]
            odd = h[:, 1::2, :]
            h = torch.stack([even + g * odd, odd + g * even], dim=2).reshape(B, N, d)
        return h

    def forward(self, x):
        B, W = x.shape
        d = self.d
        h = self.embed(x) + self.pos
        hw = h.view(B, self.Nw, self.Wl, d)
        loc = torch.stack([self._chaotic(hw[:, wi], self._sig_l, self.gates_l)
                           for wi in range(self.Nw)], dim=1)
        glob = self._chaotic(loc.mean(dim=2), self._sig_g, self.gates_g)
        gvec = glob.mean(dim=1, keepdim=True)
        last = loc[:, -1, -1, :]
        return self.readout(torch.cat([last, gvec.squeeze(1)], dim=-1))


class AttnLM(nn.Module):
    def __init__(self, V, W, d):
        super().__init__()
        self.W, self.d = W, d
        self.embed = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, V))

    def forward(self, x, use_global=True):
        B, W = x.shape
        h = self.embed(x) + self.pos
        qkv = self.qkv(h).reshape(B, W, 3, self.d).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = attn.softmax(-1)
        h = self.norm(h + self.proj(attn @ v))
        return self.readout(h[:, -1, :])


# ============ evaluate with/without prior ============
def evaluate(model, W, n_samples=20000, beta=BETA):
    model.eval()
    nll_m, nll_p, acc_m, acc_p = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, len(test_ids) - W - 1, 32):
            ctx = test_ids[i:i + W]
            y = test_ids[i + W]
            logits = model(torch.tensor(ctx, dtype=torch.long).unsqueeze(0))
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


results = {"V": V, "W": W, "W_local": W_LOCAL, "d": D_MODEL, "order": ORDER,
           "beta": BETA, "train_tokens": len(train_ids), "test_tokens": len(test_ids)}

# ============ HierChaoticLM (load exp11 checkpoint) ============
hcm = HierChaoticLM(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL)
ckpt = os.path.join(HERE, "exp11_bpe_scaling", "hier.pt")
if os.path.exists(ckpt):
    hcm.load_state_dict(torch.load(ckpt))
    print("HierChaoticLM: loaded exp11 checkpoint")
else:
    print("HierChaoticLM: no checkpoint — train from scratch")
    opt = torch.optim.Adam(hcm.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    steps = 0
    while steps < TRAIN_STEPS:
        bi = np.random.randint(0, n, size=BATCH)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
        Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
        opt.zero_grad()
        loss = lossf(hcm(X), Y)
        loss.backward()
        opt.step()
        steps += 1
# sanity check: evaluate on 5k samples, expect PPL ~25.5 (same tokenizer)
r_hcm = evaluate(hcm, W, n_samples=5000)
print(f"HierChaoticLM (sanity, 5k): {r_hcm}")
results["hierarchical"] = r_hcm

# ============ AttnLM (train fresh) ============
am = AttnLM(V, W, D_MODEL)
print(f"\nAttnLM params: {sum(p.numel() for p in am.parameters()):,}")
opt = torch.optim.Adam(am.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
t0 = time.time()
n = len(train_ids) - W - 1
steps = 0
while steps < TRAIN_STEPS:
    bi = np.random.randint(0, n, size=BATCH)
    X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
    Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
    opt.zero_grad()
    loss = lossf(am(X), Y)
    loss.backward()
    opt.step()
    steps += 1
    if steps % 4000 == 0:
        print(f"  [attn {steps:,}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
r_am = evaluate(am, W)
print(f"AttnLM: {r_am}")
results["attention"] = r_am

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)
