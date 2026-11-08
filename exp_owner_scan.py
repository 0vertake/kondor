"""Why did the clean model score 61.7% against the owner's message? Is that a false-accusation risk?

Reuses the saved probe images from stage5_transfer.py and scores them against thousands of other owners'
messages, so it trains nothing and takes minutes. It answers three questions:
  1. Is the decoder biased on generated images (does it output the same bits on every image)?
  2. Where does the real owner's score fall among other owners' scores: an unlucky draw, or something more?
  3. How many owners would be falsely accused by the full two-test verdict?

This is NOT experiment E4. Every owner here shares the same two clean models, so it measures risk across
messages for these models. E4 needs many independently fine-tuned clean models.

Usage: SHIELD_KEY=... python exp_owner_scan.py [--run px32_s2000_b32_lr0.0001] [--owners 2000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import mannwhitneyu

from shield import marking, verdict
from shield.config import CONFIGS, OUT, WORK, get_key
from shield.engine import Engine
from shield.generator import cached_sample


def scan(bits: np.ndarray, messages: np.ndarray, reference: np.ndarray | None, alpha: float, K: int = 100_000,
         seed: int = 0) -> dict:
    """Per-owner accuracy and the two-test verdict. messages: (M, m). Random-message scores are shared
    across owners, because they depend only on the decoded bits."""
    N, m = bits.shape
    ones = bits.sum(axis=0).astype(np.float64)
    gain, const = (2 * ones - N) / (N * m), (N - ones).sum() / (N * m)
    owner = messages @ gain + const
    rng = np.random.default_rng(seed)
    null = np.sort(np.concatenate([rng.integers(0, 2, (2000, m), dtype=np.int8) @ gain + const for _ in range(K // 2000)]))
    p1 = (1 + len(null) - np.searchsorted(null, owner, side="left")) / (1 + len(null))
    res = {"mean": float(owner.mean()), "std": float(owner.std()), "min": float(owner.min()), "max": float(owner.max()),
           "owners": len(messages), "test1_pass": int((p1 <= alpha).sum()), "scores": owner, "p1": p1}
    if reference is not None:
        p2 = np.array([mannwhitneyu((bits == q[None]).mean(1), (reference == q[None]).mean(1), alternative="greater").pvalue
                       for q in messages])
        res.update(test2_pass=int((p2 <= alpha).sum()), detected=int(((p1 <= alpha) & (p2 <= alpha)).sum()))
    return res


def decoder_bias(bits: np.ndarray, k: int) -> dict:
    """How fixed is the decoder's output across images?"""
    f = bits.mean(axis=0)
    sym = bits.reshape(len(bits), -1, k).sum(axis=2)
    return {"mean_fraction_of_ones": float(f.mean()), "share_of_all_zero_blocks": float((sym == 0).mean()),
            "bits_same_on_90pct_of_images": float(np.mean(np.maximum(f, 1 - f) >= 0.9))}


def public(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("scores", "p1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="px32_s2000_b32_lr0.0001")
    ap.add_argument("--owners", type=int, default=2000)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--base", default="google/ddpm-cifar10-32")
    ap.add_argument("--dataset", default="cifar10-bird")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--clean-run", default=None, help="run holding the clean model, if not this run")
    a = ap.parse_args()
    cfg = CONFIGS[a.run.split("_")[0]]
    eng = Engine.load(cfg)
    key = get_key()
    root = WORK / "e2" / a.run
    decode = lambda model: eng.decode_bits(cached_sample(model, a.n, ddim_steps=a.ddim, log=lambda s: print("  " + s, flush=True)))
    clean, base, marked = decode(WORK / "e2" / (a.clean_run or a.run) / "clean"), decode(a.base), decode(root / "marked1")
    real = marking.message_bits(key, "demo-owner", a.dataset, cfg.message_bits)
    others = np.stack([marking.message_bits(key, f"scan-owner-{i}", a.dataset, cfg.message_bits) for i in range(a.owners)])
    expected_sd = 0.5 / np.sqrt(cfg.message_bits)

    rows = {"clean model (reference: base)": (clean, base), "base model (reference: clean)": (base, clean),
            "marked model, other owners (reference: clean)": (marked, clean)}
    out = {"run": a.run, "n_images": a.n, "alpha": a.alpha, "chance_sd_for_fair_independent_bits": float(expected_sd), "models": {}}
    for name, (bits, ref) in rows.items():
        r = scan(bits, others, ref, a.alpha)
        real_score = float(verdict.bit_accuracy(bits, real).mean())
        rank = float(np.mean(r["scores"] >= real_score))
        out["models"][name] = {**public(r), "decoder_bias": decoder_bias(bits, cfg.bits_per_block),
                               "real_owner_score": real_score, "share_of_other_owners_at_or_above_real": rank}
        m = out["models"][name]
        print(f"\n[{name}]\n  decoder: {m['decoder_bias']}\n  other owners: mean {r['mean']:.4f}, SD {r['std']:.4f} "
              f"(fair coin {expected_sd:.4f}), max {r['max']:.4f}\n  real owner {real_score:.4f}; {rank:.2%} of other owners "
              f"score as high\n  test 1 passed by {r['test1_pass']}/{r['owners']}, test 2 by {r['test2_pass']}/{r['owners']}, "
              f"FALSELY DETECTED {r['detected']}/{r['owners']}", flush=True)
    path = OUT / "e2" / a.run / "owner_scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
