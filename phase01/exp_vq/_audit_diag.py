"""_audit_diag.py — внешняя проверка sts_prog (НЕ часть исследования).

Цель: отделить вклад АРХИТЕКТУРНОГО КОНТУРА от вклада ОБУЧЕНИЯ.

1) Проверить, что конфиг d=192/L=8 даёт ровно 900,353 параметров.
2) Измерить точность СЕЛЕКЦИИ (top_next == позиция B) на НЕОБУЧЕННОЙ модели.
   Контур селекции использует СЫРЫЕ эмбеддинги и не имеет обучаемых
   параметров (кроме query_proj). Если он попадает в B уже на
   инициализации -> retrieval задан архитектурой, а не выучен.
3) Аналитический FLOP-подсчёт: проверяем заявленное O(W*d*L) vs O(W^2*L).
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.join(PHASE, "exp_memory_selector"))
sys.path.append(HERE)

from experiment import load_chars, make_bpe, MAX_TRAIN, W
from models_pc import build_pc_model, count_params
from parametric_models import TransformerLM

torch.manual_seed(0)
np.random.seed(0)

print("=" * 70)
print("1) ПРОВЕРКА КОНФИГУРАЦИИ")
print("=" * 70)
m = build_pc_model("pc", vocab=512, d=192, layers=8, driver_mode="sts_prog")
n_pc = count_params(m)
tf = TransformerLM(512, 256, D=124, HEADS=4, LAYERS=4)
n_tf = count_params(tf)
print(f"  PurePCLM sts_prog d=192 L=8 : {n_pc:,}")
print(f"  TransformerLM D=124 L=4 H=4 : {n_tf:,}")
print(f"  разница параметров          : {abs(n_pc-n_tf)/n_tf*100:.2f}%")
print(f"  ГЛУБИНА: chaos blocks = {len(m.blocks)}  vs  tf layers = 4")

print()
print("=" * 70)
print("2) СЕЛЕКЦИЯ НА НЕОБУЧЕННОЙ МОДЕЛИ (архитектура без обучения)")
print("=" * 70)

train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
tok = make_bpe(train_text)
V = tok.get_vocab_size()
test_ids = np.array(tok.encode(test_text).ids, dtype=np.int32)
print(f"  V={V} test_tokens={len(test_ids):,}")

m = build_pc_model("pc", vocab=V, d=192, layers=8, driver_mode="sts_prog")
m.eval()

# ВАЖНО: нормализуем эмбеддинги так же, как после обучения? Нет — берём как есть.
# Но pos-эмбеддинги имеют std 0.02, embed — дефолт N(0,1). Оставляем инициализацию.

rng = np.random.default_rng(0)
distances = (16, 64, 128, 256)
N_TRIALS = 200
res = {}
for L in distances:
    hit_top1 = 0      # top_next[0] == позиция B
    hit_in_top8 = 0   # B есть среди top-8 соседей
    tot = 0
    for _ in range(N_TRIALS):
        found = False
        for _try in range(200):
            i = int(rng.integers(L + 2, len(test_ids) - L - 2))
            A = int(test_ids[i]); B = int(test_ids[i + 1])
            j = i + L
            if j < len(test_ids) and test_ids[j - 1] == A:
                found = True
                break
        if not found:
            continue
        window = test_ids[j - W:j]
        X = torch.tensor([window], dtype=torch.long)
        with torch.no_grad():
            _ = m(X)
        # последний блок записал last_driver_pos = top_next[:,0]
        pos = int(m.last_driver_pos[0].item())
        target_pos = W - L + 1          # позиция B внутри окна
        if pos == target_pos:
            hit_top1 += 1
        tot += 1
    res[L] = (hit_top1 / max(1, tot), tot)
    print(f"  L={L:3d}: селекция top1 попала в B = {hit_top1}/{tot} = {hit_top1/max(1,tot)*100:.1f}%")

print()
print("  (случайный уровень = 1/W = %.2f%%)" % (100.0 / W))

print()
print("=" * 70)
print("3) FLOP-БЮДЖЕТ (аналитика), W=256, d=192 (chaos) / d=124 (tf)")
print("=" * 70)
Wn = 256
dc, Lc = 192, 8
dt, Lt = 124, 4
H = 4
# --- chaos per block ---
# h @ W  : W*d^2 matmul (умножение + сложение = 2 FLOP)
chaos_matmul = 2 * Wn * dc * dc
# селекция: norm(W*d) + cosine(W*d) + topk + gather(k*d)
chaos_sel = 2 * Wn * dc + 2 * Wn * dc + 8 * dc
# sync: k*(driver-h) по всем позициям: W*d
chaos_sync = 2 * Wn * dc
# query_proj: 2*(d*d)
chaos_qp = 2 * (2 * dc * dc)
chaos_per_block = chaos_matmul + chaos_sel + chaos_sync + chaos_qp
chaos_total = Lc * chaos_per_block + 2 * (3 * dc * dc + dc * V)  # readout3
print(f"  chaos: matmul(h@W)      = {chaos_matmul/1e6:8.2f} MFLOP/блок")
print(f"         селекция+sync    = {(chaos_sel+chaos_sync)/1e6:8.2f} MFLOP/блок")
print(f"         query_proj       = {chaos_qp/1e6:8.2f} MFLOP/блок")
print(f"  ИТОГО chaos ({Lc} блоков)      = {chaos_total/1e6:8.2f} MFLOP")

# --- transformer per layer ---
tf_qkv = 3 * (2 * Wn * dt * dt)
tf_scores = 2 * Wn * Wn * dt          # QK^T  <-- квадратичный член
tf_av = 2 * Wn * Wn * dt              # attn @ V
tf_out = 2 * Wn * dt * dt
tf_ffn = 2 * (2 * Wn * dt * 4 * dt)   # 4d hidden
tf_per_layer = tf_qkv + tf_scores + tf_av + tf_out + tf_ffn
tf_total = Lt * tf_per_layer + 2 * Wn * dt * V
print(f"  tf   : QKV projections   = {tf_qkv/1e6:8.2f} MFLOP/слой")
print(f"         SCORES QK^T+@V    = {(tf_scores+tf_av)/1e6:8.2f} MFLOP/слой  <-- O(W^2)")
print(f"         out proj + FFN    = {(tf_out+tf_ffn)/1e6:8.2f} MFLOP/слой")
print(f"  ИТОГО tf    ({Lt} слоя)       = {tf_total/1e6:8.2f} MFLOP")
print()
print(f"  соотношение FLOP tf/chaos = {tf_total/chaos_total:.2f}x")
print(f"  измеренное соотношение времени 478/297 = {478/297:.2f}x")
