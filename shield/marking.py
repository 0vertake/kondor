"""Stage 2 core: owner message, marking a folder, and the hash-chained registry."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path

import numpy as np

from .config import WORK
from .engine import Engine
from .images import list_images, load_square, to_pil, to_tensor

REGISTRY = WORK / "registry.jsonl"


def message_bits(key: bytes, owner: str, dataset: str, nbits: int) -> np.ndarray:
    """First nbits of HMAC(key, owner || dataset), counter-mode extended past 256 bits."""
    out, ctr, label = b"", 0, f"{owner}\x1f{dataset}".encode()
    while len(out) * 8 < nbits:
        out += hmac.new(key, label + (b"" if ctr == 0 else b"\x1f%d" % ctr), hashlib.sha256).digest()
        ctr += 1
    return np.unpackbits(np.frombuffer(out, np.uint8))[:nbits]


def message_hash(bits: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(bits).tobytes()).hexdigest()


def _last_hash(log: Path) -> str:
    if not log.exists() or log.stat().st_size == 0:
        return "0" * 64
    return hashlib.sha256(log.read_bytes().rstrip(b"\n").rsplit(b"\n", 1)[-1]).hexdigest()


def registry_append(record: dict, log: Path = REGISTRY) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    rec = {**record, "timestamp": record.get("timestamp", time.time()), "prev_hash": _last_hash(log)}
    with open(log, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def registry_verify(log: Path = REGISTRY) -> bool:
    prev = "0" * 64
    for line in log.read_bytes().splitlines():
        if json.loads(line)["prev_hash"] != prev:
            return False
        prev = hashlib.sha256(line).hexdigest()
    return True


def registry_find(owner: str, dataset: str, log: Path = REGISTRY) -> dict | None:
    if not Path(log).exists():
        return None
    for line in Path(log).read_text().splitlines():
        r = json.loads(line)
        if r["owner"] == owner and r["dataset"] == dataset:
            return r
    return None


def mark_folder(src: Path, dst: Path, engine: Engine, key: bytes, owner: str, dataset: str,
                log: Path = REGISTRY) -> dict:
    """Resize/crop each image to the configuration size (decision D1), add W, save PNG, register."""
    bits = message_bits(key, owner, dataset, engine.cfg.message_bits)
    W = engine.pattern(bits)
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for p in list_images(src):
        out = dst / f"{p.stem}.png"
        to_pil(to_tensor(load_square(p, engine.cfg.image_size)) + W).save(out)
        hashes[out.name] = hashlib.sha256(out.read_bytes()).hexdigest()
    return registry_append({"owner": owner, "dataset": dataset, "config": engine.cfg.name,
                            "config_hash": engine.cfg.hash(), "message_hash": message_hash(bits),
                            "n_images": len(hashes),
                            "images_digest": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}, log)
