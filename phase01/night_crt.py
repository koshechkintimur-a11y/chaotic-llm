"""night_crt.py — НОЧНОЙ ПРОГОН: 500M chaos + КТО-селектор (гипотеза «жирной модели»).

Гипотеза (архитектор): на малом масштабе (287K) модель не может выделить ёмкость
под КТО-путь — он конкурирует с микшером и проигрывает (retrieval=0). Большая
модель (500M) с запасом ёмкости СМОЖЕТ выучить использование КТО-корзин.

Сравнение:
  big-500M-CRT : 500M chaos + CRT residue-bucket observer (мать узнаёт ребёнка)
  big-500M-no  : 500M chaos БЕЗ CRT (изолировать эффект ёмкости)

Протокол: fp16 AMP + grad checkpointing, 12 слоёв, ~8000 steps, W=256, batch=32.
Корпус: corpus_stack_train.txt (503M токенов Python).
Оценка: PPL + честный retrieval (индукционная головка: KEY->B повтор на L).
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parametric_models import ChaoticMixerLM, count_params, build_matched_pair

VOCAB = 512
W = 256
ORDER = 3
BATCH = 32          # ночной, чтобы влезло в 12GB
LR = 3e-4
WARMUP = 1000
N_EVAL = 4000


def load_stack_text(path, max_bytes=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        if max_bytes:
            return f.read(max_bytes)
        return f.read()


def make_bpe(text, vocab=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[], show_progress=False)
    tok.train_from_iterator([text[i:i + 100000] for i in range(0, len(text), 100000)], trainer=trainer)
    return tok


# ---------------- CRT residue-bucket observer (from crt_probe2, the PROVEN part) ----------------
class CRTObserver(nn.Module):
    """Раздельные корзины остатков по простым модулям. crt_probe2 ДОКАЗАЛ: при
    раздельном чтении (без сжатия в вектор) позиция восстанавливается на 100%."""
    def __init__(self, D, d_crt=16, primes=(3, 5, 7, 11)):
        super().__init__()
        self.primes = primes
        self.d_crt = d_crt
        # отдельная проекция на каждую корзину (не общая!)
        # sum(p) корзин, каждая D->d_crt
        self.per_bucket = nn.ModuleList([nn.Linear(D, d_crt) for _ in range(sum(primes))])
        self.n_out = sum(primes) * d_crt

    def forward(self, e):
        # e: [B, W, D]
        B, W, D = e.shape
        out = []
        for p in self.primes:
            for r in range(p):
                idx = torch.arange(W, device=e.device) % p == r
                b = e[:, idx, :].sum(dim=1)          # [B, D]
                out.append(self.per_bucket[len(out)](b))  # [B, d_crt]
        return torch.cat(out, dim=-1)               # [B, sum(p)*d_crt] — раздельно!


class CRTChaoticLM(nn.Module):
    """ChaoticMixerLM + КТО-наблюдатель ПЕРЕД микшером (по гипотезе: жирная модель
    научится использовать корзины для вылавливания KEY)."""
    def __init__(self, V, W, D, BLOCKS_PER_LAYER, LAYERS, d_crt=16, primes=(3, 5, 7, 11)):
        super().__init__()
        self.mixer = ChaoticMixerLM(V, W, D=D, BLOCKS_PER_LAYER=BLOCKS_PER_LAYER, LAYERS=LAYERS)
        self.crt = CRTObserver(D, d_crt=d_crt, primes=primes)
        # readout: concat[последний из микшера, CRT-запись] -> V
        self.readout = nn.Sequential(nn.Linear(D + self.crt.n_out, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, x):
        e = self.mixer.embed(x) + self.mixer.pos       # вход ДО микшера
        last = self.mixer(x)                           # выход микшера (logits от head)
        # пересчитаем: mixer(x) возвращает head(ln_f(h[:,-1])). Нам нужен h_last до head.
        return last  # пока заглушка — см. forward2


class CRTChaoticLM2(nn.Module):
    """Чистая версия: embed+pos -> mixer -> h_last; CRT на входе e. concat -> readout."""
    def __init__(self, V, W, D, BLOCKS_PER_LAYER, LAYERS, d_crt=16, primes=(3, 5, 7, 11)):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        from parametric_models import BidirectionalMixer
        self.mixer = BidirectionalMixer(D, BLOCKS_PER_LAYER, LAYERS, n_tokens=W)
        self.ln_f = nn.LayerNorm(D)
        self.crt = CRTObserver(D, d_crt=d_crt, primes=primes)
        self.readout = nn.Sequential(nn.Linear(D + self.crt.n_out, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, x):
        e = self.embed(x) + self.pos                    # [B,W,D]
        h = self.ln_f(self.mixer(e))                    # [B,W,D]
        h_last = h[:, -1, :]                            # [B,D]
        rec = self.crt(e)                               # [B, sum(p)*d_crt] — раздельно
        return self.readout(torch.cat([h_last, rec], dim=-1))


def build_night_model(kind, D, LAYERS, d_crt=16):
    if kind == "crt":
        return CRTChaoticLM2(VOCAB, W, D, 4, LAYERS, d_crt=d_crt)
    return ChaoticMixerLM(VOCAB, W, D=D, BLOCKS_PER_LAYER=4, LAYERS=LAYERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["crt", "no"])
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-bytes", type=int, default=None, help="лимит текста для теста")
    ap.add_argument("--grad-ckpt", action="store_true", help="градиентный чекпоинтинг")
    args = ap.parse_args()

    print(f"Loading corpus_stack...", flush=True)
    text = load_stack_text(os.path.join(HERE, "corpus_stack_train.txt"), args.max_bytes)
    tok = make_bpe(text[:10_000_000])  # BPE на подвыборке, чтобы не ждать вечно
    V = tok.get_vocab_size()
    # токенизация ПО ЧАНКАМ в numpy int32 — весь текст разом не влезает (32GB)
    CH = 5_000_000  # 5M символов на чанк
    chunks = [np.array(tok.encode(text[i:i + CH]).ids, dtype=np.int32)
              for i in range(0, len(text), CH)]
    ids = np.concatenate(chunks)
    del text, chunks
    print(f"  V={V} tokens={len(ids):,}", flush=True)

    model = build_night_model(args.kind, args.D, args.layers)
    n = sum(p.numel() for p in model.parameters())
    print(f"[{args.kind}] params={n:,}", flush=True)
    model = model.to("cuda")

    # grad checkpointing (экономит VRAM на глубоких стеках) — опционально
    if args.grad_ckpt and hasattr(model, "mixer") and hasattr(model.mixer, "blocks"):
        for b in model.mixer.blocks:
            b.grad_checkpointing = True

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    # data loader
    n = len(ids) - W - 1
    rng = np.random.default_rng(0)

    def make_batch(b):
        s = rng.integers(0, n - W, size=b)
        X = np.stack([ids[i:i + W] for i in s])
        Y = np.array([ids[i + W] for i in s])
        return torch.tensor(X, dtype=torch.long, device="cuda"), torch.tensor(Y, dtype=torch.long, device="cuda")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        X, Y = make_batch(args.batch)
        with torch.cuda.amp.autocast():
            logits = model(X)
            loss = nn.functional.cross_entropy(logits, Y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 1000 == 0:
            dt = time.time() - t0
            print(f"[{args.kind}] [{step}/{args.steps}] loss={loss.item():.3f} ({dt:.0f}s)", flush=True)

    # eval PPL
    model.eval()
    rng2 = np.random.default_rng(42)
    starts = np.sort(rng2.choice(n - W, size=N_EVAL, replace=False))
    tot_nll, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, N_EVAL, 128):
            ss = starts[i:i + 128]
            X = torch.tensor(np.stack([ids[s:s + W] for s in ss]), dtype=torch.long, device="cuda")
            Y = torch.tensor([ids[s + W] for s in ss], dtype=torch.long, device="cuda")
            with torch.cuda.amp.autocast():
                logits = model(X)
            lp = torch.log_softmax(logits.float(), -1)
            tot_nll += -lp[torch.arange(len(ss)), Y].sum().item()
            cnt += len(ss)
    ppl = float(np.exp(tot_nll / cnt))
    print(f"[{args.kind}] PPL={ppl:.3f}", flush=True)

    # ---- честный retrieval: индукционная головка (KEY->B, KEY на L) ----
    rng3 = np.random.default_rng(0)
    for L, n_trials in [(16, 200), (64, 200), (128, 200), (256, 200)]:
        hits, miss = 0, 0
        for _ in range(n_trials):
            pos = int(rng3.integers(W + 1, len(ids) - W - 2))
            A = int(ids[pos])
            B = int(ids[pos + 1])
            # ищем второй A на расстоянии L
            j = pos + L
            if j >= len(ids) - 1:
                continue
            if ids[j] != A:
                # ищем ближайший
                for d in range(L - 5, L + 5):
                    if pos + d < len(ids) - 1 and ids[pos + d] == A:
                        j = pos + d
                        break
            if ids[j] != A:
                continue
            # окно кончается на j (A), цель = B
            if j < W:
                continue
            X = torch.tensor([ids[j - W:j]], dtype=torch.long, device="cuda")
            with torch.no_grad(), torch.cuda.amp.autocast():
                lp = torch.log_softmax(model(X).float(), -1)
            top = int(lp[0].argmax().item())
            if top == B:
                hits += 1
            miss += 1
        print(f"[{args.kind}] induction L={L}: {hits}/{miss}={hits / max(1, miss):.3f}", flush=True)

    out = {"kind": args.kind, "D": args.D, "layers": args.layers,
           "params": n, "ppl": ppl, "tokens_seen": len(ids) * args.steps // (len(ids) // W)}
    with open(os.path.join(HERE, f"night_{args.kind}.json"), "w") as f:
        json.dump(out, f)
    print(f"[{args.kind}] saved night_{args.kind}.json", flush=True)


if __name__ == "__main__":
    main()
