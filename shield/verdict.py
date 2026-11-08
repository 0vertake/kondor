"""Stage 3 core: the two statistical tests and the verdict (design section 4).

Test 1, random-message comparison. Decoded bits do not depend on which message we compare with, and
the owner's message was fixed beforehand by a keyed hash. So under "no relation", the owner's score
is exchangeable with the score of a uniformly random message:
    p = (1 + #{random messages scoring >= owner}) / (1 + K)

Test 2, clean-reference comparison. Per-image bit accuracy of the suspect's images against the
owner's message versus the same quantity on a clean reference model (one-sided Mann-Whitney).
Guards against a decoder biased on one model family.

Detected only if both pass; Not detected if both fail; otherwise Inconclusive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import mannwhitneyu


def bit_accuracy(bits: np.ndarray, message: np.ndarray) -> np.ndarray:
    """Per-image bit accuracy. bits: (N, m); message: (m,)."""
    return (bits == message[None, :]).mean(axis=1)


def random_message_test(bits: np.ndarray, message: np.ndarray, K: int = 100_000, seed: int = 0,
                        chunk: int = 2000) -> dict:
    """From per-position counts of 1s, score K random messages cheaply."""
    N, m = bits.shape
    ones = bits.sum(axis=0).astype(np.float64)
    gain = (2 * ones - N) / (N * m)                  # score(r) = sum_j r_j * gain_j + const
    const = (N - ones).sum() / (N * m)
    owner = float(message @ gain + const)
    rng = np.random.default_rng(seed)
    hits, sq, mean = 0, 0.0, 0.0
    done = 0
    while done < K:
        k = min(chunk, K - done)
        s = rng.integers(0, 2, (k, m), dtype=np.int8) @ gain + const
        hits += int(np.sum(s >= owner))
        mean += float(s.sum())
        sq += float((s ** 2).sum())
        done += k
    mean /= K
    return {"owner_score": owner, "p": (1 + hits) / (1 + K), "K": K, "null_mean": mean,
            "null_std": float(np.sqrt(max(sq / K - mean ** 2, 0.0))), "z": (owner - mean) / max(np.sqrt(max(sq / K - mean ** 2, 0.0)), 1e-12)}


def reference_test(suspect_acc: np.ndarray, reference_acc: np.ndarray) -> dict:
    u = mannwhitneyu(suspect_acc, reference_acc, alternative="greater")
    return {"p": float(u.pvalue), "suspect_mean": float(suspect_acc.mean()), "reference_mean": float(reference_acc.mean()),
            "n_suspect": len(suspect_acc), "n_reference": len(reference_acc)}


def bootstrap_ci(values: np.ndarray, B: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Image-level bootstrap of the mean: patches of one image are correlated, images are the unit."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (B, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@dataclass
class Verdict:
    verdict: str
    reason: str
    bit_accuracy: float
    ci95: tuple
    test1: dict
    test2: dict | None
    n_images: int
    alpha: float

    def as_dict(self):
        return asdict(self)


def decide(bits: np.ndarray, message: np.ndarray, reference_bits: np.ndarray | None, alpha: float = 1e-3,
           min_images: int = 100, K: int = 100_000, seed: int = 0) -> Verdict:
    """alpha must be at least 1/(K+1), or test 1 can never pass."""
    if alpha < 1 / (K + 1):
        raise ValueError("alpha is below the resolution of the random-message test; raise K")
    acc = bit_accuracy(bits, message)
    t1 = random_message_test(bits, message, K=K, seed=seed)
    ci = bootstrap_ci(acc)
    t2 = reference_test(acc, bit_accuracy(reference_bits, message)) if reference_bits is not None else None
    N = len(bits)
    if N < min_images:
        return Verdict("Inconclusive", f"only {N} images (minimum {min_images})", float(acc.mean()), ci, t1, t2, N, alpha)
    if t2 is None:
        return Verdict("Inconclusive", "no clean reference outputs, so test 2 could not run", float(acc.mean()), ci, t1, t2, N, alpha)
    p1, p2 = t1["p"] <= alpha, t2["p"] <= alpha
    if p1 and p2:
        v, why = "Detected", "both tests passed"
    elif not p1 and not p2:
        v, why = "Not detected", "both tests failed; this does not mean the model is clean"
    else:
        v, why = "Inconclusive", f"tests disagree (test 1 {'pass' if p1 else 'fail'}, test 2 {'pass' if p2 else 'fail'})"
    return Verdict(v, why, float(acc.mean()), ci, t1, t2, N, alpha)
