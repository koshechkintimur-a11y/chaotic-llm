"""
train_models.py — train all models on toy tasks A–D (v2: cross-entropy, one-hot).

Usage:
  python train_models.py <task> [outdir]
    task: A, B, C, D, or "B-long"
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chaotic_torch import (ChaoticMixer, GRUModel, LocalAttn, FullAttn, MLPModel)
from toy_data import (taskA_batch, taskB_batch, taskC_batch, taskD_batch, d_in_for)
from chaos_lib import (flops_attention, flops_mlp, flops_chaotic_step,
                       flops_gsf_pooled)

torch.manual_seed(0)
np.random.seed(0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train_model(model, task, d_in, d_out, bs=64, epochs=500, lr=1e-3, extra=None,
                eval_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    history = []
    for ep in range(epochs):
        x, q, y = task(bs, **(extra or {}))
        opt.zero_grad()
        out = model(x, q)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
        if (ep + 1) % eval_every == 0:
            acc = evaluate(model, task, d_in, d_out, extra)
            history.append({"epoch": ep + 1, "loss": float(loss.item()), "acc": acc})
    return history


def evaluate(model, task, d_in, d_out, extra=None, n_batches=30, bs=128):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, q, y = task(bs, **(extra or {}))
            out = model(x, q)
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


def latency_ms(model, x, q, reps=50):
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x, q)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x, q)
        dt = (time.perf_counter() - t0) / reps * 1e3
    model.train()
    return dt


def build_models(task_name, task_N, d_in, d_out, d=32):
    return {
        "Chaotic+GSF": (
            ChaoticMixer(task_N, d_in, d_out, d_model=d, n_blocks=6, gsf_hidden=64),
            lambda n, dd=d: 6 * (flops_chaotic_step(n, dd) + flops_gsf_pooled(n, dd))),
        "Chaotic": (
            ChaoticMixer(task_N, d_in, d_out, d_model=d, n_blocks=6),
            lambda n, dd=d: 6 * flops_chaotic_step(n, dd)),
        "GRU": (
            GRUModel(task_N, d_in, d_out, d_model=d),
            lambda n, dd=d: 3 * n * dd * dd * 2 * 2),
        "LocalAttn": (
            LocalAttn(task_N, d_in, d_out, d_model=d),
            lambda n, dd=d: 2 * (flops_attention(n, dd) // 2 + flops_mlp(n, dd))),
        "FullAttn": (
            FullAttn(task_N, d_in, d_out, d_model=d, n_layers=2),
            lambda n, dd=d: 2 * (flops_attention(n, dd) + flops_mlp(n, dd))),
        "MLP": (
            MLPModel(task_N, d_in, d_out, d_model=d),
            lambda n, dd=d: flops_mlp(n, dd, hidden=64)),
    }


def run_task(task_name, task_fn, task_N, d_in, d_out, task_kw, outdir,
             pad_to=None):
    os.makedirs(outdir, exist_ok=True)
    models = build_models(task_name, task_N, d_in, d_out)
    results = {"task": task_name, "N": task_N, "models": {}}

    def wrap_task(bs, **kw):
        x, q, y = task_fn(bs, **kw)
        if pad_to is not None and x.shape[1] < pad_to:
            pad = pad_to - x.shape[1]
            x = torch.cat([x, torch.zeros(bs, pad, x.shape[2])], dim=1)
            q = torch.cat([q, torch.zeros(bs, pad)], dim=1)
        return x, q, y

    for name, (model, flops_fn) in models.items():
        t0 = time.perf_counter()
        hist = train_model(model, wrap_task, d_in, d_out, extra=task_kw)
        train_t = time.perf_counter() - t0
        acc = evaluate(model, wrap_task, d_in, d_out, task_kw, n_batches=50)
        x, q, _ = wrap_task(64, **(task_kw or {}))
        lat = latency_ms(model, x, q)
        results["models"][name] = {
            "accuracy": acc,
            "params": count_params(model),
            "flops_per_sample": flops_fn(task_N),
            "latency_ms": lat,
            "train_seconds": train_t,
            "history": hist,
        }
        print(f"[{task_name}] {name:18s} acc={acc:.3f}  params={count_params(model):6d}  "
              f"flops={flops_fn(task_N):.3e}  lat={lat:.2f}ms")
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_long_range(outdir):
    os.makedirs(outdir, exist_ok=True)
    Ls = [2, 4, 8, 16, 32, 64, 128, 256]
    max_N = 258
    N_pad = 289  # 17^2
    d_in = 16 + 1  # V=16 + marker
    d_out = 16
    all_results = {"Ls": Ls, "N_pad": N_pad, "models": {}}
    for name, (model, flops_fn) in build_models("B-long", N_pad, d_in, d_out).items():
        all_results["models"][name] = {}
        for L in Ls:
            def task_mixed(bs, L=L):
                Lr = int(L)  # train at exact L for simplicity
                x, q, y = taskB_batch(bs, Lr, N_max=max_N)
                if x.shape[1] < N_pad:
                    pad = N_pad - x.shape[1]
                    x = torch.cat([x, torch.zeros(bs, pad, x.shape[2])], dim=1)
                    q = torch.cat([q, torch.zeros(bs, pad)], dim=1)
                return x, q, y
            hist = train_model(model, task_mixed, d_in, d_out, bs=64, epochs=200,
                               eval_every=100)
            acc = evaluate(model, task_mixed, d_in, d_out, n_batches=40)
            all_results["models"][name][str(L)] = {
                "accuracy": acc,
                "flops": flops_fn(N_pad),
            }
            print(f"[B-long] {name:18s} L={L:4d} acc={acc:.3f}")
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    return all_results


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "A"
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    configs = {
        "A": {"fn": taskA_batch, "N": 16, "kw": {"N": 16},
              "d_in": d_in_for("A"), "d_out": 16,
              "dir": "exp_10_toy_transformer"},
        "B": {"fn": taskB_batch, "N": 25, "kw": {"L": 16},
              "d_in": d_in_for("B"), "d_out": 16, "pad_to": 25,
              "dir": "exp_10_toy_transformer"},
        "C": {"fn": taskC_batch, "N": 16, "kw": {"N": 16},
              "d_in": d_in_for("C"), "d_out": 32,
              "dir": "exp_10_toy_transformer"},
        "D": {"fn": taskD_batch, "N": 9, "kw": {"N": 9},
              "d_in": d_in_for("D"), "d_out": 16,
              "dir": "exp_10_toy_transformer"},
        "B-long": None,
    }
    if task == "B-long":
        run_long_range(os.path.join(base, "exp_05_long_range"))
    elif task in configs:
        cfg = configs[task]
        run_task(task, cfg["fn"], cfg["N"], cfg["d_in"], cfg["d_out"],
                 cfg["kw"], os.path.join(base, cfg["dir"]),
                 pad_to=cfg.get("pad_to"))
    else:
        print(f"unknown task: {task}")