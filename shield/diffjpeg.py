"""A differentiable JPEG, so gradients can reach the patches through compression.

Follows the JFIF steps Pillow uses (YCbCr, 4:2:0 chroma, 8x8 DCT, standard quantization tables). The only
change is the rounding, replaced by round(x) + (x - round(x))^3, which has a usable gradient. It is an
approximation: tests compare it with real Pillow JPEG.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LUMA = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_CHROMA = torch.tensor([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99]] + [[99] * 8] * 4, dtype=torch.float32)


def _dct_matrix(dev) -> torch.Tensor:
    k = torch.arange(8, dtype=torch.float32, device=dev)
    d = torch.cos(torch.pi * (2 * k[None, :] + 1) * k[:, None] / 16) * 0.5
    d[0] = d[0] / 2 ** 0.5
    return d


def _table(base: torch.Tensor, q: int, dev) -> torch.Tensor:
    s = 5000 / q if q < 50 else 200 - 2 * q
    return torch.clamp(torch.floor((base.to(dev) * s + 50) / 100), 1, 255)


def _soft_round(x: torch.Tensor) -> torch.Tensor:
    r = torch.round(x)
    return r + (x - r) ** 3


def _channel(c: torch.Tensor, table: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Compress and restore one plane (B, H, W), level-shifted. H and W must be multiples of 8."""
    B, H, W = c.shape
    b = c.reshape(B, H // 8, 8, W // 8, 8).permute(0, 1, 3, 2, 4)      # B, h, w, 8, 8
    f = D @ b @ D.T
    b = D.T @ (_soft_round(f / table) * table) @ D
    return b.permute(0, 1, 3, 2, 4).reshape(B, H, W)


def diff_jpeg(x: torch.Tensor, q: int) -> torch.Tensor:
    """x: (B, 3, H, W) in [0, 1]; H and W must be multiples of 16."""
    dev = x.device
    D = _dct_matrix(dev)
    r, g, b = (x[:, 0] * 255, x[:, 1] * 255, x[:, 2] * 255)
    y = 0.299 * r + 0.587 * g + 0.114 * b - 128
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    down = lambda c: F.avg_pool2d(c[:, None], 2)[:, 0]
    up = lambda c: F.interpolate(c[:, None], scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
    y = _channel(y, _table(_LUMA, q, dev), D) + 128
    cb = up(_channel(down(cb), _table(_CHROMA, q, dev), D))
    cr = up(_channel(down(cr), _table(_CHROMA, q, dev), D))
    out = torch.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], dim=1) / 255
    return out.clamp(0, 1)
