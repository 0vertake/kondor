"""Invisibility metrics and pattern uniformity (DiffusionShield Eq. 1)."""
from __future__ import annotations

import numpy as np
import torch


def psnr(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).numpy()
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


def ssim(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    from skimage.metrics import structural_similarity
    A, B = a.permute(0, 2, 3, 1).numpy(), b.permute(0, 2, 3, 1).numpy()
    return np.array([structural_similarity(x, y, channel_axis=2, data_range=1.0) for x, y in zip(A, B)])


def lpips_alex(a: torch.Tensor, b: torch.Tensor, batch: int = 16) -> np.ndarray:
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False)
    out = []
    with torch.no_grad():
        for i in range(0, len(a), batch):
            out.append(net(a[i:i + batch] * 2 - 1, b[i:i + batch] * 2 - 1).flatten().numpy())
    return np.concatenate(out)


def uniformity(W: torch.Tensor) -> float:
    """Z = 1 - (1/n) sum ||W_i/||W_i|| - mean||, on the patterns as actually released."""
    v = W.flatten(1).double()
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float(1 - (v - v.mean(0)).norm(dim=1).mean())
