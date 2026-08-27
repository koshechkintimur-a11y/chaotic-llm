"""exp15_learnable_beta.py — Step 5b: learnable β.

Compare:
  A) Global learnable β (scalar, sigmoid-bounded) — trained jointly.
  B) Context-dependent β_t = sigmoid(MLP(readout_state)) — per-position.

Reuses the exp14 L=1 checkpoint (mixer_1.pt, readout_1.pt) if available.
If not, trains a fresh L=1 model.
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
OUT = os.path.join(HERE, "exp15_learnable_beta")
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
TRAIN_STEPS = 12000


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


# ============ beta corpus-prior (fixed, for reference evaluation) ============
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


# ============ models ============
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


# Variant A: global learnable beta head
class BetaMixtureGlobal(nn.Module):
    """logit_beta = param, sigmoid-bounded to [0,1)."""
    def __init__(self, prior):
        super().__init__()
        self.logit_beta = nn.Parameter(torch.tensor(0.0))  # init beta=0.5
        self.prior = prior  # dict

    def forward(self, logits, ctx):
        beta = torch.sigmoid(self.logit_beta)
        logp = torch.log_softmax(logits, -1)
        cp = self._prior_logp(ctx)
        if cp is None:
            return logits  # no prior for this context
        logp_c = torch.logaddexp(torch.log1p(-beta) + logp, torch.log(beta) + torch.tensor(cp, device=logits.device))
        return logp_c

    def _prior_logp(self, ctx):
        # ctx shape (B, W) — last order tokens
        B = ctx.shape[0]
        ctx_np = ctx.cpu().numpy()
        out = np.full((B, V), -1e9, dtype=np.float64)
        for b in range(B):
            for back in range(ORDER, 0, -1):
                table = self.prior.get(tuple(ctx_np[b, -back:]))
                if table:
                    tot = sum(table.values())
                    for t, c in table.items():
                        out[b, t] = np.log(c / tot)
                    break
        return torch.tensor(out, dtype=torch.float32, device=ctx.device)


# Variant B: context-dependent beta from readout state
class BetaMixtureContext(nn.Module):
    """beta_t = sigmoid(MLP(readout_state))."""
    def __init__(self, prior, d, V):
        super().__init__()
        self.beta_net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.prior = prior
        self.V = V

    def forward(self, logits, ctx, state):
        beta = torch.sigmoid(self.beta_net(state))  # (B, 1)
        logp = torch.log_softmax(logits, -1)
        cp = self._prior_logp(ctx)
        if cp is None:
            return logits
        beta_exp = beta.expand(-1, self.V)
        logp_c = torch.logaddexp(torch.log1p(-beta_exp) + logp, torch.log(beta_exp) + cp)
        return logp_c

    def _prior_logp(self, ctx):
        B = ctx.shape[0]
        ctx_np = ctx.cpu().numpy()
        out = np.full((B, self.V), -1e9, dtype=np.float64)
        for b in range(B):
            for back in range(ORDER, 0, -1):
                table = self.prior.get(tuple(ctx_np[b, -back:]))
                if table:
                    tot = sum(table.values())
                    for t, c in table.items():
                        out[b, t] = np.log(c / tot)
                    break
        return torch.tensor(out, dtype=torch.float32, device=ctx.device)


# ============ load L=1 checkpoint ============
def build_l1_model():
    mixer = MultiLayerChaotic(V, W, W_LOCAL, D_MODEL, BLOCKS_LOCAL, BLOCKS_GLOBAL, 1)
    readout = AttnReadout(V, W, D_MODEL)
    ckpt_m = os.path.join(HERE, "exp14_multilayer", "mixer_1.pt")
    if os.path.exists(ckpt_m):
        mixer.load_state_dict(torch.load(ckpt_m, weights_only=True))
        readout.load_state_dict(torch.load(
            os.path.join(HERE, "exp14_multilayer", "readout_1.pt"), weights_only=True))
        print("loaded L=1 checkpoint from exp14")
    else:
        # train from scratch
        print("no checkpoint, training L=1 from scratch")
        opt = torch.optim.Adam(list(mixer.parameters()) + list(readout.parameters()), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        n = len(train_ids) - W - 1
        for s in range(TRAIN_STEPS):
            bi = np.random.randint(0, n, size=BATCH)
            X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
            Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
            opt.zero_grad()
            h = mixer(X)
            loss = lossf(readout(h), Y)
            loss.backward()
            opt.step()
    return mixer, readout


results = {"V": V, "W": W, "train_steps": TRAIN_STEPS, "order": ORDER}

# --- A) Global learnable beta ---
print("\n=== A) Global learnable beta ===")
mixer_a, readout_a = build_l1_model()
mixer_a.train()
readout_a.train()
beta_global = BetaMixtureGlobal(prior)
opt = torch.optim.Adam(beta_global.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
n = len(train_ids) - W - 1
for s in range(TRAIN_STEPS // 2):
    bi = np.random.randint(0, n, size=BATCH)
    X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
    Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
    with torch.no_grad():
        h = mixer_a(X)
        logits = readout_a(h)
    logp_c = beta_global(logits, X)
    loss = lossf(logp_c, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
beta_val = torch.sigmoid(beta_global.logit_beta).item()
print(f"trained beta = {beta_val:.4f}")

# evaluate
mixer_a.eval()
readout_a.eval()
nll, acc = [], 0
with torch.no_grad():
    for i in range(0, len(test_ids) - W - 1, 32):
        ctx = test_ids[i:i + W]
        y = test_ids[i + W]
        X = torch.tensor([ctx], dtype=torch.long)
        h = mixer_a(X)
        logits = readout_a(h)
        logp_c = beta_global(logits, X)
        logp = torch.log_softmax(logp_c[0], -1).numpy()
        nll.append(-logp[y])
        if int(np.argmax(logp)) == y:
            acc += 1
        if len(nll) >= 20000:
            break
n = len(nll)
r_global = {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n, "beta": beta_val}
print(f"Global learnable beta: {r_global}")
results["global_beta"] = r_global

# --- B) Context-dependent beta ---
print("\n=== B) Context-dependent beta ===")
mixer_b, readout_b = build_l1_model()
mixer_b.train()
readout_b.train()
beta_ctx = BetaMixtureContext(prior, D_MODEL, V)
opt = torch.optim.Adam(list(beta_ctx.parameters()), lr=1e-3)
lossf = nn.CrossEntropyLoss()
for s in range(TRAIN_STEPS // 2):
    bi = np.random.randint(0, n, size=BATCH)
    X = torch.tensor(np.stack([train_ids[i:i + W] for i in bi]), dtype=torch.long)
    Y = torch.tensor([train_ids[i + W] for i in bi], dtype=torch.long)
    with torch.no_grad():
        h = mixer_b(X)
        logits = readout_b(h)
    # extract state for beta
    q = readout_b.q(h[:, -1:, :])
    k = readout_b.kv(h)
    scores = (q @ k.transpose(-2, -1)) / (D_MODEL ** 0.5)
    out = readout_b.proj(scores.softmax(-1) @ h)
    state = readout_b.norm(h[:, -1:, :] + out).squeeze(1)  # (B, d)
    logp_c = beta_ctx(logits, X, state)
    loss = lossf(logp_c, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# evaluate
mixer_b.eval()
readout_b.eval()
beta_ctx.eval()
nll, acc, betas = [], 0, []
with torch.no_grad():
    for i in range(0, len(test_ids) - W - 1, 32):
        ctx = test_ids[i:i + W]
        y = test_ids[i + W]
        X = torch.tensor([ctx], dtype=torch.long)
        h = mixer_b(X)
        logits = readout_b(h)
        q = readout_b.q(h[:, -1:, :])
        k = readout_b.kv(h)
        scores = (q @ k.transpose(-2, -1)) / (D_MODEL ** 0.5)
        out = readout_b.proj(scores.softmax(-1) @ h)
        state = readout_b.norm(h[:, -1:, :] + out).squeeze(1)
        logp_c = beta_ctx(logits, X, state)
        logp = torch.log_softmax(logp_c[0], -1).numpy()
        nll.append(-logp[y])
        if int(np.argmax(logp)) == y:
            acc += 1
        betas.append(torch.sigmoid(beta_ctx.beta_net(state)).item())
        if len(nll) >= 20000:
            break
n = len(nll)
r_ctx = {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n,
         "beta_mean": float(np.mean(betas)), "beta_std": float(np.std(betas))}
print(f"Context-dependent beta: {r_ctx}")
results["context_beta"] = r_ctx

# --- C) Reference: fixed beta=0.3 (from exp13) ---
print("\n=== C) Reference: fixed beta=0.3 ===")
mixer_c, readout_c = build_l1_model()
mixer_c.eval()
readout_c.eval()
nll, acc = [], 0
with torch.no_grad():
    for i in range(0, len(test_ids) - W - 1, 32):
        ctx = test_ids[i:i + W]
        y = test_ids[i + W]
        X = torch.tensor([ctx], dtype=torch.long)
        h = mixer_c(X)
        logits = readout_c(h)
        logp = torch.log_softmax(logits[0], -1).numpy().astype(np.float64)
        cp = prior_logp(ctx)
        if cp is not None:
            logp = np.logaddexp(np.log1p(-0.3) + logp, np.log(0.3) + cp)
        nll.append(-logp[y])
        if int(np.argmax(logp)) == y:
            acc += 1
        if len(nll) >= 20000:
            break
n = len(nll)
r_fixed = {"n": n, "ppl": float(np.exp(np.mean(nll))), "acc": acc / n, "beta": 0.3}
print(f"Fixed beta=0.3: {r_fixed}")
results["fixed_beta"] = r_fixed

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved to", OUT)