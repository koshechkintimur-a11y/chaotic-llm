"""diagnose_las2.py — расширенная диагностика селекции (план архитектора, шаг 1).

Три вопроса на лучшей sts_emb конфигурации (d192_l8_T8):
  Q1. Какой % top-1 (sel) попадает в ±3 от настоящего KEY (первый A)?
  Q2. Какой % sel_next выбирает именно KEY+1 (позицию B)?
  Q3. cosine(KEY, last) после микшера — виден ли KEY запросу?

Логика теста (та же, что в induction_retrieval):
  A=test_ids[i], B=test_ids[i+1], j=i+L, test_ids[j-1]==A.
  window=[j-W:j] (последний токен = второй A).
  KEY (первый A) в окне на позиции W-1-L, B на W-L.
  Модель с sts_emb сохраняет last_driver_pos = sel_next (позиция соседа).
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_pc import build_pc_model, W

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_memory_selector"))
sys.path.insert(0, os.path.join(HERE, ".."))
from experiment import load_chars, make_bpe, MAX_TRAIN, VOCAB, W, D

def main():
    PHASE = os.path.join(HERE, "..")
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    test_ids = tok.encode(test_text).ids
    test_ids = np.array(test_ids, dtype=np.int64)

    model = build_pc_model("pc", V, d=192, alpha=0.9, k_init=1.8,
                           sync_steps=8, driver_mode="sts_emb", layers=8).to("cuda")
    model.load_state_dict(torch.load(os.path.join(HERE, "model_pc.pt")))
    model.eval()

    rng = np.random.default_rng(0)
    for L in (16, 64, 128, 256):
        q1_hit, q1_tot, q2_hit, q2_tot = 0, 0, 0, 0
        cos_key_last = []
        n = 0
        for _ in range(120):
            found = False
            for _try in range(200):
                i = int(rng.integers(L + 2, len(test_ids) - L - 2))
                A = int(test_ids[i])
                B = int(test_ids[i + 1])
                j = i + L
                if j < len(test_ids) and test_ids[j - 1] == A:
                    found = True
                    break
            if not found:
                continue
            window = test_ids[j - W:j]
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model(X)
            sel_next = int(model.last_driver_pos[0].item()) if model.last_driver_pos is not None else -1
            # KEY (первый A) в окне на W-1-L, B на W-L
            key_pos = W - 1 - L
            b_pos = W - L
            # Q1: sel должен быть на KEY (тогда sel_next = KEY+1 = B)
            # проверяем: попадает ли sel (после вычитания 1) в ±3 от KEY
            sel = sel_next - 1  # позиция, которую выбрали как A
            if abs(sel - key_pos) <= 3:
                q1_hit += 1
            q1_tot += 1
            # Q2: sel_next == позиция B
            if sel_next == b_pos:
                q2_hit += 1
            q2_tot += 1
            # Q3: cosine(KEY, last) из скрытых состояний (нужен доступ к h)
            n += 1
        print(f"L={L}: Q1(sel~KEY±3)={q1_hit/max(1,q1_tot):.3f}  "
              f"Q2(sel_next==B)={q2_hit/max(1,q2_tot):.3f}")

if __name__ == "__main__":
    main()
