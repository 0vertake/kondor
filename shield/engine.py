"""Stage 1 core: the patch bank and the decoder, trained jointly (DiffusionShield, Algorithm 2).

Each step: (a) add a random patch to random image blocks and train the decoder with SGD to say
which patch it was; (b) with the decoder fixed, update patches 2..B with 5 PGD steps inside the
l-infinity budget, so they become easier to recognise. Patch 1 stays zero (paper, Sec. 3.4).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ENGINE_VERSION, WORK, EngineConfig, device


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                                 nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))

    def forward(self, x):
        return F.relu(x + self.net(x))


class Decoder(nn.Module):
    """Initial conv, two 64-filter residual blocks, global average pool (paper, App. C.3)."""

    def __init__(self, B: int, ch: int = 64):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(ResBlock(ch), ResBlock(ch))
        self.fc = nn.Linear(ch, B)

    def forward(self, x):
        return self.fc(self.blocks(self.head(x)).mean(dim=(2, 3)))


def random_blocks(images: torch.Tensor, n: int, p: int, gen: torch.Generator) -> torch.Tensor:
    """n random p x p crops from (N, 3, H, W) images in [0, 1]."""
    N, _, H, W = images.shape
    idx = torch.randint(0, N, (n,), generator=gen)
    ys = torch.randint(0, H - p + 1, (n,), generator=gen)
    xs = torch.randint(0, W - p + 1, (n,), generator=gen)
    return torch.stack([images[i, :, y:y + p, x:x + p] for i, y, x in zip(idx.tolist(), ys.tolist(), xs.tolist())])


def train_engine(cfg: EngineConfig, images: torch.Tensor, log: Callable[[str], None] = print,
                 steps: int | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    steps = steps or cfg.steps
    eps, alpha = cfg.eps, cfg.eps / 10
    dev = device()
    patches = ((torch.rand(cfg.B, 3, cfg.patch, cfg.patch, generator=gen) * 2 - 1) * eps).to(dev)
    patches[0] = 0
    dec = Decoder(cfg.B).to(dev)
    opt = torch.optim.SGD(dec.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    t0 = time.time()
    hist = []
    for step in range(steps):
        p = random_blocks(images, cfg.batch, cfg.patch, gen).to(dev)
        y = torch.randint(0, cfg.B, (cfg.batch,), generator=gen).to(dev)
        # (a) decoder step
        dec.train()
        logits = dec((p + patches[y]).clamp(0, 1))
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (logits.argmax(1) == y).float().mean().item()
        # (b) patch step: 5-step PGD with the decoder held fixed
        dec.eval()
        for q in dec.parameters():
            q.requires_grad_(False)
        for _ in range(cfg.pgd_steps):
            w = patches.clone().requires_grad_(True)
            F.cross_entropy(dec((p + w[y]).clamp(0, 1)), y).backward()
            with torch.no_grad():
                patches = (patches - alpha * w.grad.sign()).clamp(-eps, eps)
                patches[0] = 0
        for q in dec.parameters():
            q.requires_grad_(True)
        if step % 100 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss), "acc": acc, "seconds": time.time() - t0})
            log(f"[engine {cfg.name}] step {step}/{steps} loss {float(loss):.4f} block acc {acc:.3f} ({time.time() - t0:.0f}s)")

    # Marks are applied to 8-bit images: snap patches to whole levels, then let the decoder adapt.
    patches = (patches * 255).round() / 255
    for _ in range(min(300, steps // 5)):
        p = random_blocks(images, cfg.batch, cfg.patch, gen).to(dev)
        y = torch.randint(0, cfg.B, (cfg.batch,), generator=gen).to(dev)
        dec.train()
        x = ((p + patches[y]).clamp(0, 1) * 255).round() / 255
        loss = F.cross_entropy(dec(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    dec.eval().cpu()                       # decoding is cheap: checkpoints and audits stay on CPU
    patches = patches.cpu()
    return {"config": cfg, "patches": patches, "decoder": dec, "history": hist, "train_seconds": time.time() - t0}


def checkpoint_path(cfg: EngineConfig) -> Path:
    return WORK / "engines" / f"{cfg.name}-{cfg.hash()}.pt"


def save_engine(res: dict, extra: dict | None = None) -> Path:
    cfg = res["config"]
    path = checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"version": ENGINE_VERSION, "config_hash": cfg.hash(), "config": cfg.__dict__, "train_seconds": res["train_seconds"],
            "history": res["history"], **(extra or {})}
    torch.save({"patches": res["patches"], "decoder": res["decoder"].state_dict(), "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


class Engine:
    """A trained configuration: renders a message into a pattern and reads blocks back."""

    def __init__(self, cfg: EngineConfig, patches: torch.Tensor, decoder: Decoder):
        self.cfg, self.patches, self.decoder = cfg, patches, decoder.eval()

    @classmethod
    def load(cls, cfg: EngineConfig, path: Path | None = None) -> "Engine":
        ck = torch.load(path or checkpoint_path(cfg), weights_only=True)
        dec = Decoder(cfg.B)
        dec.load_state_dict(ck["decoder"])
        return cls(cfg, ck["patches"], dec)

    def symbols_from_bits(self, bits: np.ndarray) -> np.ndarray:
        k = self.cfg.bits_per_block
        assert len(bits) == self.cfg.message_bits
        return (bits.reshape(-1, k) * (1 << np.arange(k - 1, -1, -1))).sum(axis=1)

    def bits_from_symbols(self, sym: np.ndarray) -> np.ndarray:
        k = self.cfg.bits_per_block
        return ((sym[..., None] >> np.arange(k - 1, -1, -1)) & 1).reshape(*sym.shape[:-1], -1).astype(np.uint8)

    def pattern(self, bits: np.ndarray) -> torch.Tensor:
        """The tiled watermark W (3, S, S) for a message."""
        n = self.cfg.image_size // self.cfg.patch
        sym = torch.from_numpy(self.symbols_from_bits(bits).astype(np.int64))
        rows = self.patches[sym].reshape(n, n, 3, self.cfg.patch, self.cfg.patch)
        return torch.cat([torch.cat(list(r), dim=2) for r in rows], dim=1)

    @torch.no_grad()
    def decode_bits(self, images: torch.Tensor, batch: int = 2048) -> np.ndarray:
        """(N, 3, S, S) in [0, 1] -> (N, message_bits) decoded bits."""
        p, N = self.cfg.patch, images.shape[0]
        n = self.cfg.image_size // p
        x = images[:, :, :n * p, :n * p].unfold(2, p, p).unfold(3, p, p)     # N,3,n,n,p,p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, p, p)
        sym = torch.cat([self.decoder(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)]).numpy()
        return self.bits_from_symbols(sym.reshape(N, n * n))
