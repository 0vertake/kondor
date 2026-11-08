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
    k = min(k, int(len(x) ** 0.5))
    img = to_pil(torch.cat([torch.cat(list(x[i * k:(i + 1) * k]), dim=2) for i in range(k)], dim=1))
    img.resize((img.width * 3, img.height * 3), Image.Resampling.NEAREST).save(path)


def share_curve(results: dict, path):
    """E3's curve: bit accuracy and verdict against the marked share of the training set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(((float(k), v) for k, v in results["shares"].items()), key=lambda r: r[0])
    xs, ys = [r[0] * 100 for r in rows], [r[1]["bit_accuracy"] for r in rows]
    lo, hi = [r[1]["ci95"][0] for r in rows], [r[1]["ci95"][1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], marker="o", capsize=3, color="tab:blue")
    for x, y, (_, v) in zip(xs, ys, rows):
        ax.annotate(v["verdict"], (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.axhline(results["clean_model"]["bit_accuracy"], color="tab:grey", ls="--", label="clean model")
    ax.set_xscale("log")
    ax.set_xlabel("marked share of the training set (%)")
    ax.set_ylabel("bit accuracy on generated images")
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"{results['dataset']}, {results['config']}, {results['steps']} steps, {results['n']} images per model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


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
    ap.add_argument("--pre-jpeg", type=int, default=None,
                    help="save the training images (clean and marked) as JPEG at this quality before fine-tuning")
    ap.add_argument("--reuse-clean", default=None, help="run name whose clean model to reuse (same data and settings)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = CONFIGS[a.config]
    a.base = a.base or {32: "google/ddpm-cifar10-32", 256: "google/ddpm-celebahq-256"}[cfg.image_size]
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
    if a.pre_jpeg:
        # The product's real case: images are compressed after marking and before anyone trains on them.
        if a.reuse_clean:
            raise SystemExit("--pre-jpeg changes the clean training images, so the clean model cannot be reused")
        from shield.robust import _real_jpeg
        squash = lambda x: torch.cat([_real_jpeg(x[i:i + 512], a.pre_jpeg) for i in range(0, len(x), 512)])
        clean, marked = squash(clean), squash(marked)
    jq = f"_jpeg{a.pre_jpeg}" if a.pre_jpeg else ""
    run = f"{cfg.name}_s{a.steps}_b{a.batch}_lr{a.lr:g}{jq}{a.tag}"
    root = WORK / "e2" / run
    out = OUT / "e2" / run
    out.mkdir(parents=True, exist_ok=True)

    def get_samples(name: str, images: torch.Tensor) -> torch.Tensor:
        info = fine_tune(a.base, images, root / name, a.steps, a.batch, a.lr)
        print(f"[stage5] {name}: fine-tuned, {info['seconds']:.0f}s, loss {info['final_loss']:.4f}", flush=True)
        return cached_sample(root / name, a.n, ddim_steps=a.ddim, batch=a.sample_batch, log=lambda s: print("  " + s, flush=True))

    if a.reuse_clean:
        Xc = cached_sample(WORK / "e2" / a.reuse_clean / "clean", a.n, ddim_steps=a.ddim, batch=a.sample_batch)
    else:
        Xc = get_samples("clean", clean)
    grid(Xc, out / "clean_samples.png")
    bc = eng.decode_bits(Xc)
    bb = eng.decode_bits(cached_sample(a.base, a.n, ddim_steps=a.ddim, batch=a.sample_batch, log=lambda s: print("  " + s, flush=True)))
    clean_v = verdict.decide(bc, msg, bb).as_dict()
    print(f"[E2] clean model vs base: {clean_v['verdict']}; bit acc {clean_v['bit_accuracy']:.4f}; p1 {clean_v['test1']['p']:.2e}; "
          f"p2 {clean_v['test2']['p']:.2e}", flush=True)
    results = {"base": a.base, "dataset": dataset, "pre_jpeg": a.pre_jpeg, "steps": a.steps, "batch": a.batch, "lr": a.lr, "n": a.n, "ddim": a.ddim,
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
        bm = eng.decode_bits(Xm)
        v = verdict.decide(bm, msg, bc).as_dict()
        # A low share might mark only some outputs: a large spread between images would show that
        v["per_image_sd"] = float(verdict.bit_accuracy(bm, msg).std())
        results["shares"][str(share)] = v
        print(f"[E2] share {share:g}: {v['verdict']}; bit acc {v['bit_accuracy']:.4f} CI {v['ci95']}; p1 {v['test1']['p']:.2e}; "
              f"p2 {v['test2']['p']:.2e}; clean model {clean_v['bit_accuracy']:.4f}", flush=True)
    (out / "e2.json").write_text(json.dumps(results, indent=2, default=float))
    if len(results["shares"]) > 1:
        share_curve(results, out / "e3_curve.png")
        print("\n[E3] share   verdict        bit acc   per-image SD   p test 1   p test 2")
        for k, v in sorted(results["shares"].items(), key=lambda r: -float(r[0])):
            print(f"[E3] {float(k):5.0%}   {v['verdict']:13s}  {v['bit_accuracy']:.4f}    {v['per_image_sd']:.4f}         "
                  f"{v['test1']['p']:.2e}   {v['test2']['p']:.2e}", flush=True)
    print(f"[stage5] -> {out}")


if __name__ == "__main__":
    main()
