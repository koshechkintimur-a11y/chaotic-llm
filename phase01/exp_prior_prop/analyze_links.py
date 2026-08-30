"""analyze_links.py — кривая accuracy(L) + шум-тест (ТЗ раздел 4).

Loads a trained model, measures retrieval accuracy vs distance L and
noise-robustness, writes plots + json.

Usage: python analyze_links.py DP [--n-trials 300]
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(HERE)
REPO = os.path.dirname(PHASE)
for p in (HERE, PHASE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import build_model, VOCAB, W
from exp_memory_selector.experiment import load_chars, build_tokenizer, MAX_TRAIN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    tok = build_tokenizer()
    V = tok.get_vocab_size()
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    return tok, V, tok.encode(train_text).ids, tok.encode(test_text).ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["DP", "DP-noprop", "DP-rand", "C-cap", "PM", "SP"])
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--noise-frac", type=float, default=0.9)
    args = ap.parse_args()

    tok, V, train_ids, test_ids = load_data()
    model = build_model(args.config, vocab=V).to(DEVICE)
    # load weights if saved separately? results json has ppl only; retrain not needed:
    # this script reads results_*.json metrics and adds the LINK analysis by
    # re-running the model — so we require the model to be trained.
    # For now: load state if a .pt exists, else instruct.
    pt = os.path.join(HERE, f"model_{args.config}.pt")
    if os.path.exists(pt):
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        print(f"loaded {pt}", flush=True)
    else:
        print(f"WARNING: {pt} not found — model is randomly initialized. "
              f"Results will be noise. Retrain or add checkpoint saving.",
              flush=True)

    model.eval()
    from collections import Counter
    tr_counts = Counter(train_ids)
    cands = [t for t in range(150, V) if tr_counts.get(t, 0) >= 20]
    rng = np.random.default_rng(7)

    # --- accuracy vs distance ---
    distances = (16, 64, 256, 1024)
    retr = {}
    for L in distances:
        acc = 0.0
        n_t = 0
        for _ in range(args.n_trials):
            key = int(rng.choice(cands))
            i = int(rng.integers(W + L, len(test_ids) - 1))
            ctx = list(test_ids[i - W:i])
            key_pos = W - L
            if key_pos >= 0:
                ctx[key_pos] = key
            n_noise = int(W * args.noise_frac)
            noise_idx = rng.choice(W, size=n_noise, replace=False)
            for ni in noise_idx:
                if ni == key_pos:
                    continue
                ctx[ni] = int(rng.integers(1, 100))
            with torch.no_grad():
                logits = model(torch.tensor([ctx], dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
            n_t += 1
            if int(np.argmax(logits)) == key:
                acc += 1.0
        retr[L] = acc / max(1, n_t)
        print(f"  L={L}: acc={retr[L]:.4f}", flush=True)

    # --- noise robustness at fixed L=64 ---
    noise_res = {}
    for nf in (0.0, 0.95, 0.99, 0.995):
        acc = 0.0
        n_t = 0
        for _ in range(args.n_trials):
            key = int(rng.choice(cands))
            i = int(rng.integers(W + 64, len(test_ids) - 1))
            ctx = list(test_ids[i - W:i])
            ctx[W - 64] = key
            n_noise = int(W * nf)
            noise_idx = rng.choice(W, size=n_noise, replace=False)
            for ni in noise_idx:
                if ni == W - 64:
                    continue
                ctx[ni] = int(rng.integers(1, 100))
            with torch.no_grad():
                logits = model(torch.tensor([ctx], dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
            n_t += 1
            if int(np.argmax(logits)) == key:
                acc += 1.0
        noise_res[nf] = acc / max(1, n_t)
        print(f"  noise={nf}: acc={noise_res[nf]:.4f}", flush=True)

    out = {"config": args.config, "retrieval": retr, "noise": noise_res}
    json.dump(out, open(os.path.join(HERE, f"links_{args.config}.json"), "w"), indent=2)
    print(f"saved links_{args.config}.json", flush=True)


if __name__ == "__main__":
    main()
