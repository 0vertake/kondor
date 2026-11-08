"""Image loading and the fixed preprocessing used everywhere (design decision D1: resize + crop)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESAMPLE = Image.Resampling.BICUBIC


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def load_square(path: Path, size: int) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    im = im.crop((l, t, l + s, t + s))
    return im if s == size else im.resize((size, size), RESAMPLE)


def to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im.convert("RGB"), np.float32).copy() / 255.0).permute(2, 0, 1)


def to_pil(x: torch.Tensor) -> Image.Image:
    return Image.fromarray((x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())


def folder_tensor(folder: Path, size: int, limit: int | None = None) -> torch.Tensor:
    ps = list_images(folder)[:limit]
    return torch.stack([to_tensor(load_square(p, size)) for p in ps])


def cifar10(root: Path, train: bool = True, cls: int | None = None) -> torch.Tensor:
    """CIFAR-10 as (N, 3, 32, 32) in [0, 1], optionally one class."""
    from torchvision.datasets import CIFAR10
    d = CIFAR10(root=str(root), train=train, download=True)
    x = torch.from_numpy(np.asarray(d.data, np.float32).copy() / 255.0).permute(0, 3, 1, 2)
    if cls is not None:
        x = x[torch.tensor(d.targets) == cls]
    return x
