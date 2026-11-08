"""STAGE 1: train the watermark engine once, then run experiment E1.

Achieves: a bank of faint patches and a small decoder that recognises them. E1 asks whether the mark
is invisible and survives normal image handling. Pass line, fixed before running: bit accuracy >= 90%
on JPEG 80 and on resize. No generator is involved yet.

Usage: python stage1_engine.py --config px32 [--steps N] [--n-test 1000]
       python stage1_engine.py --config px256 --photos data/photos
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shield import marking, metrics, verdict
from shield.attacks import ATTACKS, apply_batch
from shield.config import CONFIGS, OUT, WORK
from shield.engine import Engine, checkpoint_path, save_engine, train_engine
from shield.images import cifar10, folder_tensor, to_pil

KEY_E1 = b"e1-public-test-key"     # E1 only measures the channel; no real owner key is involved


def load_data(a, cfg):
    """Engine training crops and held-out E1 images never overlap."""
    if cfg.image_size == 32:
        root = WORK / "cifar"
        return cifar10(root, train=True), cifar10(root, train=False)[:a.n_test], "CIFAR-10 train / test"
    x = folder_tensor(a.photos, cfg.image_size)
    n_train = max(int(len(x) * 0.5), len(x) - a.n_test)
    return x[:n_train], x[n_train:], f"{a.photos}: {n_train} train / {len(x) - n_train} test"


def side_by_side(x, m, path: Path, k: int = 6):
    from PIL import Image
    d = ((m[:k] - x[:k]) * 10 + 0.5).clamp(0, 1)
    rows = [torch.cat(list(t), dim=2) for t in (x[:k], m[:k], d)]
    img = to_pil(torch.cat(rows, dim=1))
    if x.shape[-1] < 128:
        img = img.resize((img.width * 4, img.height * 4), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="px32", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--photos", type=Path, default=Path("data/photos"))
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = CONFIGS[a.config]
    train, test, source = load_data(a, cfg)
    print(f"[stage1] {cfg.name}: {source}; {cfg.message_bits}-bit message, eps {cfg.eps255}/255", flush=True)
    if a.retrain or not checkpoint_path(cfg).exists():
        res = train_engine(cfg, train, steps=a.steps)
        path = save_engine(res, {"training_images": source})
        print(f"[stage1] trained in {res['train_seconds']:.0f}s -> {path}", flush=True)
    eng = Engine.load(cfg)

    # E1: one owner's message on held-out images
    msg = marking.message_bits(KEY_E1, "e1-owner", "e1-dataset", cfg.message_bits)
    W = eng.pattern(msg)
    x = (test * 255).round() / 255
    m = ((x + W).clamp(0, 1) * 255).round() / 255
    out = OUT / "e1" / cfg.name
    side_by_side(x, m, out / "side_by_side.png")
    res = {"config": cfg.name, "config_hash": cfg.hash(), "n_images": len(x), "source": source, "message_bits": cfg.message_bits,
           "eps255": cfg.eps255, "linf_levels": float(((m - x).abs().max() * 255).round()),
           "psnr_mean": float(metrics.psnr(x, m).mean()), "ssim_mean": float(metrics.ssim(x, m).mean()),
           "lpips_mean": float(metrics.lpips_alex(x, m).mean()), "uniformity_Z": metrics.uniformity(m - x),
           "attacks": {}}
    other = marking.message_bits(KEY_E1, "e1-other-owner", "e1-dataset", cfg.message_bits)
    for name in ATTACKS:
        am, ax = apply_batch(m, name), apply_batch(x, name)
        acc = verdict.bit_accuracy(eng.decode_bits(am), msg)
        res["attacks"][name] = {"bit_accuracy": float(acc.mean()), "ci95": verdict.bootstrap_ci(acc),
                                "wrong_owner_accuracy": float(verdict.bit_accuracy(eng.decode_bits(am), other).mean()),
                                "unmarked_accuracy": float(verdict.bit_accuracy(eng.decode_bits(ax), msg).mean()),
                                "attacked_psnr": float(metrics.psnr(x, am).mean())}
        r = res["attacks"][name]
        print(f"[E1] {name:14s} bit acc {r['bit_accuracy']:.4f}  wrong owner {r['wrong_owner_accuracy']:.4f}  "
              f"unmarked {r['unmarked_accuracy']:.4f}  PSNR {r['attacked_psnr']:.1f}", flush=True)
    res["pass"] = bool(res["attacks"]["jpeg80"]["bit_accuracy"] >= 0.9 and res["attacks"]["resize50"]["bit_accuracy"] >= 0.9)
    (out / "e1.json").write_text(json.dumps(res, indent=2))
    print(f"[E1] PSNR {res['psnr_mean']:.1f} dB, SSIM {res['ssim_mean']:.4f}, LPIPS {res['lpips_mean']:.5f}, "
          f"uniformity Z {res['uniformity_Z']:.3f}; PASS={res['pass']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
