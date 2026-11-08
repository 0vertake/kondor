"""Robust engine training: DiffusionShield's joint optimization with corruptions added (paper, Sec. 4.6).

The paper says corruptions are applied after the patches are added, without giving a recipe. JPEG works on
whole images (its own 8x8 grid, coarser chroma), so unlike plain training this tiles patches onto whole
images, corrupts them, and then reads blocks. Each batch draws one corruption, as in MBRS (Jia et al. 2021):
real JPEG teaches the decoder the true compression, simulated JPEG carries gradients to the patches.

Deliberately NOT trained on: WebP and cropping, so E1 also shows how far this generalizes.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import EngineConfig, device
from .diffjpeg import diff_jpeg
from .engine import Decoder

KINDS = ["none", "jpeg_real", "jpeg_sim", "resize", "blur", "noise"]
WEIGHTS = torch.tensor([0.15, 0.30, 0.20, 0.20, 0.10, 0.05])
JPEG_Q = (50, 95)
RESIZE = (0.5, 0.9)
BLUR_SIGMA = (0.3, 1.0)
NOISE_SIGMA = (0.0, 5 / 255)


@dataclass
class Corruption:
    kind: str
    q: int = 80
    f: float = 0.75
    sigma: float = 0.5


def draw(gen: torch.Generator) -> Corruption:
    u = torch.rand(3, generator=gen).tolist()
    kind = KINDS[int(torch.multinomial(WEIGHTS, 1, generator=gen))]
    sig = NOISE_SIGMA if kind == "noise" else BLUR_SIGMA
    return Corruption(kind, int(JPEG_Q[0] + u[0] * (JPEG_Q[1] - JPEG_Q[0])), RESIZE[0] + u[1] * (RESIZE[1] - RESIZE[0]),
                      sig[0] + u[2] * (sig[1] - sig[0]))


def _real_jpeg(x: torch.Tensor, q: int) -> torch.Tensor:
    arr = (x.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out = []
    for a in arr:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=q)
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255)
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).to(x.device)


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma + 0.5))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)


def corrupt(x: torch.Tensor, c: Corruption, differentiable: bool, gen: torch.Generator) -> torch.Tensor:
    """differentiable=True swaps real JPEG for the simulation, so the patch update gets gradients."""
    if c.kind == "none":
        return x
    if c.kind == "jpeg_real" and not differentiable:
        return _real_jpeg(x, c.q)
    if c.kind in ("jpeg_real", "jpeg_sim"):
        return diff_jpeg(x, c.q)
    if c.kind == "resize":
        S = x.shape[-1]
        small = F.interpolate(x, size=max(2, round(S * c.f)), mode="bicubic", antialias=True, align_corners=False)
        return F.interpolate(small, size=S, mode="bicubic", align_corners=False).clamp(0, 1)
    if c.kind == "blur":
        return _blur(x, max(c.sigma, 0.05)).clamp(0, 1)
    return (x + c.sigma * torch.randn(x.shape, generator=gen).to(x.device)).clamp(0, 1)


def tile(patches: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """patches (B, 3, p, p), sym (N, n, n) -> patterns (N, 3, n*p, n*p). Differentiable in patches."""
    N, n, _ = sym.shape
    p = patches.shape[-1]
    return patches[sym].permute(0, 3, 1, 4, 2, 5).reshape(N, 3, n * p, n * p)


def blocks_of(x: torch.Tensor, p: int) -> torch.Tensor:
    return x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, p, p)


def train_robust_engine(cfg: EngineConfig, images: torch.Tensor, log: Callable[[str], None] = print, steps: int | None = None,
                        images_per_step: int = 32) -> dict:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    steps = steps or cfg.steps
    dev = device()
    eps, alpha = cfg.eps, cfg.eps / 10
    n = cfg.image_size // cfg.patch
    patches = ((torch.rand(cfg.B, 3, cfg.patch, cfg.patch, generator=gen) * 2 - 1) * eps).to(dev)
    patches[0] = 0
    dec = Decoder(cfg.B).to(dev)
    opt = torch.optim.SGD(dec.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    finish = min(300, steps // 5)
    t0, hist, acc_by_kind = time.time(), [], {k: [] for k in KINDS}

    def batch():
        idx = torch.randint(0, len(images), (images_per_step,), generator=gen)
        sym = torch.randint(0, cfg.B, (images_per_step, n, n), generator=gen).to(dev)
        pick = torch.randperm(images_per_step * n * n, generator=gen)[:cfg.batch].to(dev)
        return images[idx].to(dev), sym, pick

    for step in range(steps + finish):
        X, sym, pick = batch()
        c = draw(gen)
        y = sym.reshape(-1)[pick]
        freeze = step >= steps
        if step == steps:                       # marks are applied to 8-bit images: snap to whole levels
            patches = (patches * 255).round() / 255
        # (a) decoder step, seeing the true corruption
        dec.train()
        x = (X + tile(patches, sym)).clamp(0, 1)
        logits = dec(blocks_of(corrupt(x, c, False, gen), cfg.patch)[pick])
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc_by_kind[c.kind].append((logits.argmax(1) == y).float().mean().item())
        # (b) patch step: 5-step PGD through the differentiable version of the same corruption
        if not freeze:
            dec.eval()
            for q in dec.parameters():
                q.requires_grad_(False)
            for _ in range(cfg.pgd_steps):
                w = patches.clone().requires_grad_(True)
                xp = (X + tile(w, sym)).clamp(0, 1)
                F.cross_entropy(dec(blocks_of(corrupt(xp, c, True, gen), cfg.patch)[pick]), y).backward()
                with torch.no_grad():
                    patches = (patches - alpha * w.grad.sign()).clamp(-eps, eps)
                    patches[0] = 0
            for q in dec.parameters():
                q.requires_grad_(True)
        if step % 200 == 0 or step == steps + finish - 1:
            recent = {k: round(sum(v[-50:]) / len(v[-50:]), 3) for k, v in acc_by_kind.items() if v}
            hist.append({"step": step, "loss": loss.item(), "block_acc": recent, "seconds": time.time() - t0})
            log(f"[robust {cfg.name}] step {step}/{steps + finish} loss {loss.item():.3f} block acc {recent} ({time.time() - t0:.0f}s)")
    dec.eval().cpu()
    return {"config": cfg, "patches": patches.cpu(), "decoder": dec, "history": hist, "train_seconds": time.time() - t0}
