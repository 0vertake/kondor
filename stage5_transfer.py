"""STAGE 5: the decisive test. Does the mark survive training? (Experiments E2 and E3, reduced.)

Achieves: fine-tunes the same generator twice, on clean and on marked images, samples from both with
the same seeds, and asks the detector whether the marked model's outputs carry the owner's message.
With --shares it repeats at lower marked shares (E3).

Default: a REDUCED E2 on google/ddpm-cifar10-32 at 32 px, which a CPU can run. A pass supports the claim at
32 px only. --config px256 runs the design's E2 on google/ddpm-celebahq-256 and needs a GPU (about 7 days on
the laptop CPU) plus --images, a folder of training images whose license allows the experiment (D6).

Usage: SHIELD_KEY=... python stage5_transfer.py --steps 2000 --n 500 --shares 1.0
       SHIELD_KEY=... python stage5_transfer.py --config px256 --images faces/ --batch 4 --sample-batch 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shield import marking, metrics, verdict
from shield.config import CONFIGS, OUT, WORK, get_key
from shield.engine import Engine
from shield.generator import cached_sample, fine_tune
from shield.images import cifar10, folder_tensor, to_pil

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def grid(x: torch.Tensor, path, k: int = 8):
    from PIL import Image
    img = to_pil(torch.cat([torch.cat(list(x[i * k:(i + 1) * k]), dim=2) for i in range(k)], dim=1))
    img.resize((img.width * 3, img.height * 3), Image.Resampling.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="px32", choices=list(CONFIGS))
    ap.add_argument("--images", type=Path, default=None, help="training images (required for px256)")
    ap.add_argument("--base", default=None)
    ap.add_argument("--sample-batch", type=int, default=32)
    ap.add_argument("--cls", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--shares", type=float, nargs="+", default=[1.0])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = CONFIGS[a.config]
    a.base = a.base or {"px32": "google/ddpm-cifar10-32", "px256": "google/ddpm-celebahq-256"}[a.config]
    eng = Engine.load(cfg)
    if a.images:
        dataset, clean = a.images.name, folder_tensor(a.images, cfg.image_size)
    elif cfg.image_size == 32:
        dataset, clean = f"cifar10-{CLASSES[a.cls]}", cifar10(WORK / "cifar", train=True, cls=a.cls)
    else:
        raise SystemExit("--images is required for this configuration")
    owner = "demo-owner"
    msg = marking.message_bits(get_key(), owner, dataset, cfg.message_bits)
    clean = (clean * 255).round() / 255
    W = eng.pattern(msg)
    marked = ((clean + W).clamp(0, 1) * 255).round() / 255
    run = f"{cfg.name}_s{a.steps}_b{a.batch}_lr{a.lr:g}{a.tag}"
    root = WORK / "e2" / run
    out = OUT / "e2" / run
    out.mkdir(parents=True, exist_ok=True)

    def get_samples(name: str, images: torch.Tensor) -> torch.Tensor:
        info = fine_tune(a.base, images, root / name, a.steps, a.batch, a.lr)
        print(f"[stage5] {name}: fine-tuned, {info['seconds']:.0f}s, loss {info['final_loss']:.4f}", flush=True)
        return cached_sample(root / name, a.n, ddim_steps=a.ddim, batch=a.sample_batch, log=lambda s: print("  " + s, flush=True))

    Xc = get_samples("clean", clean)
    grid(Xc, out / "clean_samples.png")
    bc = eng.decode_bits(Xc)
    bb = eng.decode_bits(cached_sample(a.base, a.n, ddim_steps=a.ddim, batch=a.sample_batch, log=lambda s: print("  " + s, flush=True)))
    clean_v = verdict.decide(bc, msg, bb).as_dict()
    print(f"[E2] clean model vs base: {clean_v['verdict']}; bit acc {clean_v['bit_accuracy']:.4f}; p1 {clean_v['test1']['p']:.2e}; "
          f"p2 {clean_v['test2']['p']:.2e}", flush=True)
    results = {"base": a.base, "dataset": dataset, "steps": a.steps, "batch": a.batch, "lr": a.lr, "n": a.n, "ddim": a.ddim,
               "config": cfg.name, "config_hash": cfg.hash(), "uniformity_Z_training_set": metrics.uniformity(marked - clean),
               "clean_model": clean_v, "base_model_bit_accuracy": float(verdict.bit_accuracy(bb, msg).mean()), "shares": {}}
    # What a perfect generator would give: decoder accuracy on the marked training images themselves
    results["decoder_on_marked_training_images"] = float(verdict.bit_accuracy(eng.decode_bits(marked[:1000]), msg).mean())
    for share in a.shares:
        g = torch.Generator().manual_seed(1)
        keep = torch.randperm(len(clean), generator=g)[: round(share * len(clean))]
        imgs = clean.clone()
        imgs[keep] = marked[keep]
        Xm = get_samples(f"marked{share:g}", imgs)
        grid(Xm, out / f"marked{share:g}_samples.png")
        v = verdict.decide(eng.decode_bits(Xm), msg, bc).as_dict()
        results["shares"][str(share)] = v
        print(f"[E2] share {share:g}: {v['verdict']}; bit acc {v['bit_accuracy']:.4f} CI {v['ci95']}; p1 {v['test1']['p']:.2e}; "
              f"p2 {v['test2']['p']:.2e}; clean model {clean_v['bit_accuracy']:.4f}", flush=True)
    (out / "e2.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[stage5] -> {out}")


if __name__ == "__main__":
    main()
