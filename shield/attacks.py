"""The transforms of experiment E1: what a dataset goes through on its way to a trainer."""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from .images import RESAMPLE, to_pil, to_tensor


def _codec(im: Image.Image, fmt: str, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, fmt, quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _rescale(im: Image.Image, f: float) -> Image.Image:
    w, h = im.size
    return im.resize((round(w * f), round(h * f)), RESAMPLE).resize((w, h), RESAMPLE)


def _center_crop(im: Image.Image, keep: float) -> Image.Image:
    """Keep the central area and scale back up: this moves every block off the decoder's grid."""
    w, h = im.size
    cw, ch = round(w * keep), round(h * keep)
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch)).resize((w, h), RESAMPLE)


ATTACKS = {
    "none": lambda im: im,
    "jpeg90": lambda im: _codec(im, "JPEG", 90),
    "jpeg80": lambda im: _codec(im, "JPEG", 80),
    "jpeg60": lambda im: _codec(im, "JPEG", 60),
    "webp80": lambda im: _codec(im, "WEBP", 80),
    "resize50": lambda im: _rescale(im, 0.5),
    "center_crop90": lambda im: _center_crop(im, 0.9),
}


def apply_batch(x: torch.Tensor, name: str) -> torch.Tensor:
    fn = ATTACKS[name]
    return torch.stack([to_tensor(fn(to_pil(i))) for i in x])
