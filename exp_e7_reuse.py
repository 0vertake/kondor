"""E7: does one trained configuration serve different owners and image domains without retraining?

Pass line: per-owner decoding >= 90% bit accuracy, with cross-owner confusion at chance. Domains: CIFAR-10
test images, and real photos (Unsplash) downscaled to 32 px, which the engine never saw.

Usage: python exp_e7_reuse.py --photos ../kondor-images/data/photos
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shield import marking, verdict
from shield.config import CONFIGS, OUT, WORK
from shield.engine import Engine
from shield.images import cifar10, folder_tensor

KEY = b"e7-public-test-key"   # measures the channel only; no real owner key involved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--owners", type=int, default=20)
    a = ap.parse_args()
    cfg = CONFIGS["px32"]
    eng = Engine.load(cfg)
    domains = {"cifar10_test": cifar10(WORK / "cifar", train=False)[:1000], "photos_32px": folder_tensor(a.photos, 32)}
    msgs = [marking.message_bits(KEY, f"owner{i}", "d", cfg.message_bits) for i in range(a.owners)]
    res = {"n_owners": a.owners, "domains": {}}
    for name, x in domains.items():
        x = (x * 255).round() / 255
        own, cross = [], []
        for i, m in enumerate(msgs):
            bits = eng.decode_bits(((x + eng.pattern(m)).clamp(0, 1) * 255).round() / 255)
            own.append(float(verdict.bit_accuracy(bits, m).mean()))
            cross.append(float(verdict.bit_accuracy(bits, msgs[(i + 1) % a.owners]).mean()))
        res["domains"][name] = {"n_images": len(x), "own_mean": sum(own) / len(own), "own_min": min(own),
                                "cross_mean": sum(cross) / len(cross), "cross_max": max(cross),
                                "unmarked_mean": float(verdict.bit_accuracy(eng.decode_bits(x), msgs[0]).mean())}
        print(name, res["domains"][name], flush=True)
    res["pass"] = all(d["own_min"] >= 0.9 and abs(d["cross_mean"] - 0.5) < 0.05 for d in res["domains"].values())
    (OUT / "e7").mkdir(parents=True, exist_ok=True)
    (OUT / "e7" / "e7.json").write_text(json.dumps(res, indent=2))
    print("PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()
