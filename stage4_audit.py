"""STAGE 4: audit suspects end to end, with no manual steps (experiment E5's pipeline).

Achieves: takes crawler records and, for each, checks eligibility, generates probe images from the
suspect and its claimed base model with the same seeds, decodes, decides, and writes an evidence report.
Anything we cannot test is logged 'Not testable' with the reason, never as clean.

Usage: SHIELD_KEY=... python stage4_audit.py suspects.jsonl --owner acme --dataset gallery --config px32 --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shield import intake, marking, report, verdict
from shield.config import CONFIGS, OUT, get_key
from shield.engine import Engine
from shield.generator import cached_sample


def null_scores(bits: np.ndarray, K: int = 20_000) -> np.ndarray:
    N, m = bits.shape
    ones = bits.sum(0).astype(float)
    r = np.random.default_rng(0).integers(0, 2, (K, m), dtype=np.int8)
    return (r @ ((2 * ones - N) / (N * m))) + (N - ones).sum() / (N * m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suspects", type=Path)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="px32", choices=list(CONFIGS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=OUT / "audits")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    eng = Engine.load(cfg)
    msg = marking.message_bits(get_key(), a.owner, a.dataset, cfg.message_bits)
    reg = marking.registry_find(a.owner, a.dataset)
    if reg is None or reg["message_hash"] != marking.message_hash(msg):
        raise SystemExit("owner/dataset not registered, or this key does not reproduce the registered message")
    if not marking.registry_verify():
        raise SystemExit("registry hash chain is broken")
    rows = []
    for s in intake.read_suspects(a.suspects):
        ok, why = intake.eligibility(s)
        if not ok:
            rows.append({"suspect": s.suspect_id, "verdict": "Not testable", "reason": why})
            print(f"[audit] {s.suspect_id}: Not testable ({why})", flush=True)
            continue
        print(f"[audit] {s.suspect_id}: probing", flush=True)
        log = lambda m: print("  " + m, flush=True)
        X = cached_sample(s.source_url, a.n, a.seed, a.ddim, log)
        R = cached_sample(s.claimed_base_model, a.n, a.seed, a.ddim, log)
        bx, br = eng.decode_bits(X), eng.decode_bits(R)
        v = verdict.decide(bx, msg, br, alpha=a.alpha).as_dict()
        probe = {"n": a.n, "seed": a.seed, "ddim_steps": a.ddim, "image_hash": report.image_hashes(X)}
        path = report.write_report(a.out / s.suspect_id, s.as_dict(), reg, cfg, v, probe, X, bx, br, msg, null_scores(bx))
        rows.append({"suspect": s.suspect_id, "verdict": v["verdict"], "reason": v["reason"], "bit_accuracy": v["bit_accuracy"],
                     "p_test1": v["test1"]["p"], "p_test2": v["test2"]["p"], "report": str(path)})
        print(f"[audit] {s.suspect_id}: {v['verdict']} (bit acc {v['bit_accuracy']:.4f}) -> {path}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
