"""exp51_triple_mixer.py — Gears: three mixers (64 → 128 → 256).

Exp51 of the "gears" hypothesis. See ТЗ / chaotic_gears.py.
Note: ТЗ window3=1024 exceeds base W=256 (kept from exp41b for
apples-to-apples) — adapted to 64/128/256, same three-scale idea.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gear_protocol import run

if __name__ == "__main__":
    run(kind="triple", name="exp51-triple",
        out_dir="exp51_triple_mixer",
        blocks=8, window1=64, window2=128, window3=256,
        stride1=4, stride2=8)
