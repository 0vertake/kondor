"""Stage 4 core: the evidence report, generated with no manual steps."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

LIMITS = ("A 'Detected' verdict supports one thing: this model's outputs reproduce a pattern that only the owner's "
          "marked images are known to carry. It does not show intent, who trained the model, or how the images were "
          "obtained; a third party who redistributed the marked images would trigger the same result. 'Not detected' "
          "does not mean the model is clean. Removal and forgery attacks on this watermark have not been tested.")


def image_hashes(x: torch.Tensor) -> str:
    h = hashlib.sha256()
    for i in x:
        h.update((i * 255).round().byte().numpy().tobytes())
    return h.hexdigest()


def decoded_map(bits: np.ndarray, message: np.ndarray, n: int, k: int) -> np.ndarray:
    """Per-block correctness for one image: 255 correct, 0 wrong, upscaled for viewing."""
    ok = (bits == message).reshape(n * n, k).all(axis=1).reshape(n, n)
    return np.kron(ok.astype(np.uint8) * 255, np.ones((16 * 8 // n + 1, 16 * 8 // n + 1), np.uint8))


def write_report(out: Path, suspect: dict, registry: dict, cfg, verdict: dict, probe: dict,
                 suspect_imgs: torch.Tensor, suspect_bits: np.ndarray, reference_bits: np.ndarray | None,
                 message: np.ndarray, null_scores: np.ndarray) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from .images import to_pil
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].hist(null_scores, bins=60, color="tab:grey")
    axs[0].axvline(verdict["test1"]["owner_score"], color="tab:red", label="owner's message")
    axs[0].set_title("Test 1: random messages vs owner")
    axs[0].set_xlabel("mean bit accuracy over images")
    axs[0].legend()
    acc = (suspect_bits == message[None]).mean(axis=1)
    axs[1].hist(acc, bins=30, alpha=0.6, label="suspect", color="tab:red")
    if reference_bits is not None:
        axs[1].hist((reference_bits == message[None]).mean(axis=1), bins=30, alpha=0.6, label="clean reference", color="tab:grey")
    axs[1].set_title("Test 2: per-image bit accuracy")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(out / "nulls.png", dpi=120)
    plt.close(fig)

    n, k = cfg.image_size // cfg.patch, cfg.bits_per_block
    tiles = []
    for i in range(min(6, len(suspect_imgs))):
        im = np.asarray(to_pil(suspect_imgs[i]).resize((128, 128), Image.Resampling.NEAREST))
        mp = np.asarray(Image.fromarray(decoded_map(suspect_bits[i], message, n, k)).resize((128, 128), Image.Resampling.NEAREST).convert("RGB"))
        tiles.append(np.vstack([im, mp]))
    Image.fromarray(np.hstack(tiles)).save(out / "examples.png")

    t1, t2 = verdict["test1"], verdict["test2"]
    L = [f"# Evidence report: {suspect['suspect_id']}", "",
         f"**Verdict: {verdict['verdict']}** ({verdict['reason']}).", "",
         "| Item | Value |", "|---|---|",
         f"| Owner / dataset | {registry['owner']} / {registry['dataset']} |",
         f"| Registry record | message hash `{registry['message_hash'][:16]}…`, chained to `{registry['prev_hash'][:16]}…` |",
         f"| Configuration | {cfg.name} (`{cfg.hash()}`), {cfg.image_size} px, {cfg.patch}x{cfg.patch} patches, B={cfg.B}, budget {cfg.eps255:g}/255 |",
         f"| Suspect | {suspect['source_url']} ({suspect['family']}), claimed base `{suspect['claimed_base_model']}` |",
         f"| Probe | {probe['n']} images, seeds from {probe['seed']}, DDIM {probe['ddim_steps']} steps |",
         f"| Image set SHA-256 | `{probe['image_hash']}` |",
         f"| Bit accuracy (owner's message) | {verdict['bit_accuracy']:.4f} (95% CI {verdict['ci95'][0]:.4f}–{verdict['ci95'][1]:.4f}, image-level bootstrap) |",
         f"| Test 1, random messages | p = {t1['p']:.2e} over K = {t1['K']:,}; {t1['z']:.1f} SD above the null (resolution limit {1 / (t1['K'] + 1):.1e}) |"]
    if t2:
        L.append(f"| Test 2, clean reference | p = {t2['p']:.2e}; suspect {t2['suspect_mean']:.4f} vs reference {t2['reference_mean']:.4f} |")
    L += [f"| Threshold per test | alpha = {verdict['alpha']:g}; false-accusation rate NOT yet validated on many clean models (E4) |", "",
          "![Nulls](nulls.png)", "", "Examples, top row images, bottom row decoded blocks (white = block matches the owner's message):", "",
          "![Examples](examples.png)", "", "## Limits", "", LIMITS, "", "## Software", "",
          f"Python {sys.version.split()[0]}, torch {torch.__version__}, numpy {np.__version__}, {platform.platform()}."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "verdict.json").write_text(json.dumps({"suspect": suspect, "verdict": verdict, "probe": probe}, indent=2, default=float))
    return out / "report.md"
