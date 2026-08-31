"""diagnose_las.py — диагностика: кодируется ли KEY в скрытых состояниях PC-миксера,
и выбирает ли top1-селекция правильную позицию (KEY на дистанции L).

Вопрос ТЗ: если retrieval не растёт при правильной селекции — проблема глубже
(как KEY кодируется в скрытых состояниях до синхронизации). Измеряем:
  1. cosine(h[KEY_pos], h[last]) — похожи ли состояния KEY и последней позиции
  2. куда top1-селекция указывает относительно KEY_pos
  3. точность: выбирается ли позиция KEY+1 (содержит B)
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp_memory_selector"))

from experiment import load_chars, make_bpe, MAX_TRAIN, VOCAB, W, D, BLOCKS as B
from models_pc import build_pc_model

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")


def main():
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    test_ids = tok.encode(test_text).ids

    model = build_pc_model("pc", V, k_init=1.8, sync_steps=4, driver_mode="top1").to("cuda")
    model.load_state_dict(torch.load(os.path.join(HERE, "model_pc.pt")))
    model.eval()

    rng = np.random.default_rng(0)
    for L in (16, 64, 128, 256):
        n_trials = 300
        cos_vals = []
        drv_rel = []
        sel_key1 = 0
        sel_near_key = 0
        for _ in range(n_trials):
            i = int(rng.integers(L + 2, len(test_ids) - L - 3))
            j = i + L
            window = test_ids[j - W:j]
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad():
                h = model.embed(X) + model.pos
                for blk in model.blocks:
                    drv = model._select_driver(h)
                    h = blk(h, drv)
            key_pos = W - L
            h_key = h[0, key_pos]
            h_last = h[0, -1]
            cos_vals.append(torch.cosine_similarity(h_key, h_last, dim=0).item())
            # настоящий cosine всех позиций с последней (без маски)
            qn = h[0, -1] / (h[0, -1].norm() + 1e-6)
            hn = h[0] / (h[0].norm(dim=-1, keepdim=True) + 1e-6)
            sim2 = (hn * qn).sum(-1)
            sel = int(sim2[:W - 8].argmax())  # без последних 8 (как в модели)
            drv_rel.append(sel - key_pos)
            if sel == key_pos + 1:
                sel_key1 += 1
            if abs(sel - key_pos) <= 3:
                sel_near_key += 1
        print(f"L={L}: cos(KEY,last)={np.mean(cos_vals):.3f}±{np.std(cos_vals):.3f} | "
              f"top1 отн. KEY={np.mean(drv_rel):+.1f} | "
              f"в KEY+1={sel_key1/n_trials:.3f} | в ±3={sel_near_key/n_trials:.3f} "
              f"(rand={7/256:.3f})")

    print("Интерпретация:")
    print("  cos(KEY,last) высокий (>0.5) -> KEY похож на запрос -> селекция может его найти")
    print("  top1 в ±3 от KEY >> rand -> селекция находит KEY -> проблема в чтении соседа B")
    print("  top1 в ±3 ≈ rand -> KEY не кодируется различимо -> проблема в кодировании")


if __name__ == "__main__":
    main()
