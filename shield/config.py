"""Engine configurations. A configuration is trained once and shared by all owners."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Point these at Google Drive on Colab so results survive a disconnected session.
WORK = Path(os.environ.get("SHIELD_WORK", ROOT / "work"))
OUT = Path(os.environ.get("SHIELD_OUT", ROOT / "out"))
ENGINE_VERSION = "ds-v1"


@dataclass(frozen=True)
class EngineConfig:
    name: str
    image_size: int
    patch: int            # block side length (paper: u = v)
    B: int                # symbols per block; each block carries log2(B) bits
    eps255: float = 4.0   # l-infinity budget in 8-bit levels
    steps: int = 3000     # joint-optimization batches (paper gives no epoch count)
    batch: int = 256
    lr: float = 0.01      # SGD, momentum 0.9, weight decay 5e-4 (paper, App. C.3)
    pgd_steps: int = 5    # step size = eps / 10 (paper, App. C.3)
    seed: int = 7

    @property
    def blocks(self) -> int:
        return (self.image_size // self.patch) ** 2

    @property
    def bits_per_block(self) -> int:
        return {2: 1, 4: 2, 8: 3, 16: 4}[self.B]

    @property
    def message_bits(self) -> int:
        return self.blocks * self.bits_per_block

    @property
    def eps(self) -> float:
        return self.eps255 / 255.0

    def hash(self) -> str:
        blob = json.dumps({"v": ENGINE_VERSION, **asdict(self)}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# Paper's examples: 4x4/B=4 at 32 px (128 bits); 16x16/B=2 at 256 px (256 bits).
# eps = 4/255: the paper's 1/255 and 2/255 did not transfer at 256 px (about 54% bit accuracy).
CONFIGS = {
    "px32": EngineConfig("px32", 32, 4, 4),
    "px256": EngineConfig("px256", 256, 16, 2, batch=128),
}


def get_key() -> bytes:
    k = os.environ.get("SHIELD_KEY")
    if not k:
        raise SystemExit("SHIELD_KEY is not set (export SHIELD_KEY=<secret>).")
    return k.encode()


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
