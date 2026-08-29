"""noise_retrieval.py — secondary test: does Eye help on KEY-in-noise retrieval?

Trains each config on the synthetic task and measures retrieval accuracy.
Task: [noise(1..10)... KEY(100..511) ...noise] -> next token == KEY.
A model that routes info to the KEY position gets high accuracy.

Usage: python noise_retrieval.py <E0|E1_u|E1_l|E1_r>
"""
import os, sys, json, time, argparse
import torch
import numpy as np
from models import (EmbedMix, LMHead, BidirectionalMixer, D, W, V, R, K, DEVICE,
                    make_batch, EYE_K)

import importlib.util
spec = importlib.util.spec_from_file_location("expmod", os.path.join(os.path.dirname(__file__), "experiment.py"))
expmod = importlib.util.module_from_spec(spec)
# do not run training code at import: guard with __name__ check is in experiment.py main()


def build(form, mode):
    from models import EyeModulatorLM, BaselineLM
    if form == "E0":
        return BaselineLM()
    return EyeModulatorLM(mode=mode)


def train_noise(model, steps=4000, noise=0.95):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    N = 64
    for s in range(steps):
        Xb, Yb = make_batch(N, W, V, noise)
        logits = model(Xb)
        loss = lossf(logits.view(-1, V), Yb.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_noise(model, n=200, noise=0.95):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            Xb, Yb = make_batch(1, W, V, noise)
            logits = model(Xb)[0]
            pred = logits.argmax(-1).item()
            if pred == Yb[0].item(): correct += 1
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    form, mode = (args.config.split("_") + [None])[:2] if "_" in args.config else (args.config, None)
    model = build(form, mode).to(DEVICE)
    t = time.time()
    model = train_noise(model)
    acc = eval_noise(model)
    print(f"[{args.config}] retrieval_acc={acc:.3f} ({time.time()-t:.0f}s)")
    out = os.path.join(os.path.dirname(__file__), "noise_retrieval.json")
    data = json.load(open(out)) if os.path.exists(out) else {}
    data[args.config] = acc
    json.dump(data, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()