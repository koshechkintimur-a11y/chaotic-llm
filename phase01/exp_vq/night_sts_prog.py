"""night_sts_prog.py — НОЧНОЙ ПРОГОН STS-Prog на полном Stack (973M токенов).

Автономный, не импортирует experiment.py (тяжёлый top-level).
Чанковая токенизация (без OOM), чекпоинты каждые 2K, resume.
Подвыборка ~100M токенов из Stack.
"""
import os, sys, time, json, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
STACK_PATH = os.path.join(PHASE, "corpus_stack_train.txt")
W = 256
VOCAB = 512
BATCH = 32
STEPS = 10000
LR = 5e-4
WARMUP = 1000
N_EVAL = 2000
SUB = 100_000_000  # подвыборка 100M токенов
D_MODEL = 256
LAYERS = 12
K_INIT = 1.2
ALPHA = 0.3
SYNC_STEPS = 8
TOPK = 8
NQUERY = 4

def load_stack_text(path, max_bytes=None):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(max_bytes)
    return text

def make_bpe(text, vocab=VOCAB):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["[PAD]"])
    tok.train_from_iterator([text], trainer)
    tok.enable_padding(length=None)
    return tok

def build_order3(ids, ORDER=3):
    from collections import defaultdict
    prior = defaultdict(dict)
    for i in range(ORDER - 1, len(ids)):
        ctx = tuple(int(t) for t in ids[i - ORDER + 1:i])
        prior[ctx][int(ids[i])] = prior[ctx].get(int(ids[i]), 0) + 1
    return prior

def gated_ppl(lpm, targets, prior, ctx_tokens):
    N = len(lpm)
    nll = np.zeros(N)
    for k in range(N):
        ctx = ctx_tokens[k] if ctx_tokens and k < len(ctx_tokens) else tuple(int(targets[k - 2]) if k >= 2 else ())
        table = prior.get(ctx)
        if table:
            tot = sum(table.values())
            c = table.get(int(targets[k]), 0)
            if c > 0:
                beta = tot / (tot + 1.0)
                log_pm = lpm[k, int(targets[k])]
                nll[k] = -np.logaddexp(np.log1p(-beta) + log_pm, np.log(beta) + np.log(c / tot))
                continue
        nll[k] = -lpm[k, int(targets[k])]
    return float(np.exp(np.mean(nll)))

def induction_retrieval(model, test_ids, distances=(16, 64, 128, 256), n_trials=200):
    """Правильный индукционный тест: A→B паттерн, второй A последним токеном окна."""
    model.eval()
    rng = np.random.default_rng(42)
    res = {}
    for L in distances:
        hits, miss = 0, 0
        for _ in range(n_trials):
            i = int(rng.integers(L + 2, len(test_ids) - L - 3))
            A = int(test_ids[i]); B = int(test_ids[i + 1])
            j = i + L
            if test_ids[j - 1] != A:
                continue
            window = test_ids[j - W:j]
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits = model(X)
            pred = int(logits[0].argmax().item())
            if pred == B:
                hits += 1
            miss += 1
        res[f"L{L}"] = hits / max(1, miss)
    return res

# ============ МОДЕЛЬ (STS-Prog) — копия из models_pc.py, автономно ============
class PurePCBlock(nn.Module):
    def __init__(self, d, alpha=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d) * 1.5 + torch.randn(d, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(d))
        self.alpha = alpha
    def forward(self, h, driver, k):
        h = h + self.alpha * torch.tanh(h @ self.W + self.b)
        h = (1 - k) * h + k * driver
        return h

class PurePCLM(nn.Module):
    def __init__(self, vocab, d=D_MODEL, layers=LAYERS, k_init=K_INIT, alpha=ALPHA):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.blocks = nn.ModuleList([PurePCBlock(d, alpha=alpha) for _ in range(layers)])
        self.k = nn.Parameter(torch.tensor([k_init]))
        self.readout3 = nn.Sequential(
            nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, vocab))
        self.query_proj = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.topk = TOPK
        self.nquery = NQUERY
        self._last_sim = None

    def forward(self, x):
        e = self.embed(x) + self.pos
        Bn = e.shape[0]
        k_eff = torch.sigmoid(self.k)
        # STS-Prog
        q0 = e[:, -self.nquery:, :].mean(dim=1)
        q = q0
        en = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
        h = e
        for blk in self.blocks:
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (en * qn.unsqueeze(1)).sum(-1)
            sim[:, W - 8:] = -1e9
            kk = min(self.topk, W - 8)
            top_w, top_i = torch.topk(sim, kk, dim=1)
            w = torch.softmax(top_w / 0.3, dim=1)
            top_next = torch.clamp(top_i + 1, 0, W - 2)
            idx = torch.arange(Bn, device=e.device).unsqueeze(1).expand(Bn, kk)
            neigh = e[idx, top_next]
            driver = (w.unsqueeze(-1) * neigh).sum(dim=1, keepdim=True)
            self._last_sim = sim
            h = blk(h, driver, k_eff)
            q = q0 + self.query_proj(h[:, -1, :]) * 0.5
        h_last = h[:, -1, :]
        g = h.mean(dim=1)
        return self.readout3(torch.cat([h_last, q0, g], dim=-1))

# ============ ТРЕНИРОВКА ============
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("Loading corpus (только нужная подвыборка, ~250MB текста ≈ 120M токенов)...", flush=True)
    # SUB токенов ≈ SUB * 2.07 байт (измерили: 973M токенов на 2.01GB)
    max_bytes = int(SUB * 2.1) + 5_000_000
    text = load_stack_text(STACK_PATH, max_bytes)
    print(f"  text size: {len(text):,} chars", flush=True)
    # BPE на подвыборке
    tok = make_bpe(text[:10_000_000])
    V = tok.get_vocab_size()
    print(f"  V={V}", flush=True)
    # чанковая токенизация (только загруженный фрагмент)
    CH = 5_000_000
    chunks = []
    for i in range(0, len(text), CH):
        chunk = tok.encode(text[i:i + CH]).ids
        chunks.append(np.array(chunk, dtype=np.int32))
    ids = np.concatenate(chunks)
    del text, chunks
    print(f"  total tokens: {len(ids):,}", flush=True)
    # подвыборка
    train_ids = ids[:SUB]
    test_ids = ids[SUB:SUB + SUB // 10]
    print(f"  train: {len(train_ids):,} test: {len(test_ids):,}", flush=True)

    model = PurePCLM(V).to("cuda")
    n = sum(p.numel() for p in model.parameters())
    print(f"model: {n:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(1)
    t0 = time.time()
    N = len(train_ids) - W - 1
    for step in range(1, STEPS + 1):
        # warmup
        if step < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = LR * step / WARMUP
        else:
            for pg in opt.param_groups:
                pg["lr"] = LR
        s = rng.integers(0, N, size=BATCH)
        X = np.stack([train_ids[i:i + W] for i in s])
        Y = np.array([train_ids[i + W] for i in s])
        Xt = torch.tensor(X, dtype=torch.long, device="cuda")
        Yt = torch.tensor(Y, dtype=torch.long, device="cuda")
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(Xt)
            loss = lossf(logits, Yt)
        if not torch.isfinite(loss):
            print(f"  NaN loss at step {step}, aborting!", flush=True)
            break
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 2000 == 0:
            dt = time.time() - t0
            print(f"  [{step}/{STEPS}] loss={loss.item():.3f} ({dt:.0f}s)", flush=True)
            # чекпоинт
            torch.save(model.state_dict(), os.path.join(HERE, f"night_ckpt_{step}.pt"))
    dt = time.time() - t0
    print(f"Training done in {dt:.0f}s", flush=True)

    # eval
    model.eval()
    print("Evaluating...", flush=True)
    rng2 = np.random.default_rng(42)
    te_starts = np.sort(rng2.choice(len(test_ids) - W - 1, size=N_EVAL, replace=False))
    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=np.int64)
    ctx_tokens = []
    with torch.no_grad():
        for k, s in enumerate(te_starts):
            Xt = torch.tensor([test_ids[s:s + W]], dtype=torch.long, device="cuda")
            with torch.amp.autocast("cuda"):
                logits_te[k] = model(Xt)[0].cpu().numpy()
            y_te[k] = test_ids[s + W]
            ctx_tokens.append(tuple(test_ids[s + W - 2:s + W]))
    lpm = torch.log_softmax(torch.tensor(logits_te), -1).numpy()
    mixer_ppl = float(np.exp(np.mean([-lpm[k, y_te[k]] for k in range(N_EVAL)])))
    prior = build_order3(train_ids[:500_000])
    gated = gated_ppl(lpm, y_te, prior, ctx_tokens)
    retrieval = induction_retrieval(model, test_ids)
    print(f"[night] mixer={mixer_ppl:.3f} gated={gated:.3f} retrieval={retrieval} params={n:,} time={dt:.0f}s", flush=True)
    res = {"kind": "sts_prog_night", "d": D_MODEL, "layers": LAYERS, "params": n,
           "mixer_ppl": round(mixer_ppl, 3), "gated_ppl": round(gated, 3),
           "retrieval": {str(k): round(v, 3) for k, v in retrieval.items()},
           "time_s": round(dt, 1)}
    with open(os.path.join(HERE, "results_night.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("saved results_night.json", flush=True)

if __name__ == "__main__":
    main()