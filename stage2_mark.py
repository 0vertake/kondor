"""STAGE 2: protect a dataset.

Achieves: an owner's folder of images becomes marked PNGs carrying that owner's secret message, and a
tamper-evident registry record proves what was marked and when. No retraining per owner.

Usage: SHIELD_KEY=... python stage2_mark.py <src_folder> <dst_folder> --owner acme --dataset gallery --config px32
"""
from __future__ import annotations

import argparse
from pathlib import Path

from shield import marking
from shield.config import CONFIGS, get_key
from shield.engine import Engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="px32", choices=list(CONFIGS))
    a = ap.parse_args()
    rec = marking.mark_folder(a.src, a.dst, Engine.load(CONFIGS[a.config]), get_key(), a.owner, a.dataset)
    print(f"marked {rec['n_images']} images -> {a.dst}; registry chain valid: {marking.registry_verify()}")


if __name__ == "__main__":
    main()
