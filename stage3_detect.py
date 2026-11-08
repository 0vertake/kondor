"""STAGE 3: read the mark and decide.

Achieves: given images from a suspect and from a clean reference, returns a calibrated verdict
(Detected / Not detected / Inconclusive) from two independent tests.

Usage: SHIELD_KEY=... python stage3_detect.py <suspect_imgs> <reference_imgs> --owner acme --dataset gallery --config px32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield import marking, verdict
from shield.config import CONFIGS, get_key
from shield.engine import Engine
from shield.images import folder_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suspect", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="px32", choices=list(CONFIGS))
    ap.add_argument("--alpha", type=float, default=1e-3)
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    eng = Engine.load(cfg)
    msg = marking.message_bits(get_key(), a.owner, a.dataset, cfg.message_bits)
    v = verdict.decide(eng.decode_bits(folder_tensor(a.suspect, cfg.image_size)), msg,
                       eng.decode_bits(folder_tensor(a.reference, cfg.image_size)), alpha=a.alpha)
    print(json.dumps(v.as_dict(), indent=2, default=float))


if __name__ == "__main__":
    main()
