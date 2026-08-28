"""exp50_dual_mixer.py — Gears: two mixers (local 64 + intermediate 256).

Exp50 of the "gears" hypothesis. See ТЗ / chaotic_gears.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gear_protocol import run

if __name__ == "__main__":
    run(kind="dual", name="exp50-dual",
        out_dir="exp50_dual_mixer",
        blocks=12, window1=64, window2=256, stride=4)
