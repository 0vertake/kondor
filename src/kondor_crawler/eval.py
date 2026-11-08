"""Ranker eval: proxy Precision@K and hit-rate@K from probe_results."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from kondor_crawler.candidate import ModelCandidate


def precision_at_k(
    ranked: list[ModelCandidate],
    labels: dict[str, str],
    k: int,
    *,
    positive: str = "worthy",
    negative: str = "skip",
) -> tuple[float | None, int]:
    top = ranked[:k]
    decisive = [c for c in top if labels.get(c.uid) in {positive, negative}]
    if not decisive:
        return None, 0
    tp = sum(1 for c in decisive if labels[c.uid] == positive)
    return tp / len(decisive), len(decisive)


def average_precision(
    ranked: list[ModelCandidate],
    labels: dict[str, str],
    *,
    positive: str = "worthy",
) -> float | None:
    pos_total = sum(1 for lab in labels.values() if lab == positive)
    if not pos_total:
        return None
    hits = 0
    s = 0.0
    for i, c in enumerate(ranked, 1):
        if labels.get(c.uid) == positive:
            hits += 1
            s += hits / i
    return s / pos_total


def hit_rate_at_k(ranked_uids: Iterable[str], hits: set[str], k: int) -> float | None:
    top = list(ranked_uids)[:k]
    if not top:
        return None
    return sum(1 for u in top if u in hits) / len(top)


def summarize(
    ranked: list[ModelCandidate],
    *,
    labels: dict[str, str] | None = None,
    hits: set[str] | None = None,
) -> dict:
    out: dict = {"n": len(ranked)}
    if labels:
        out["labels"] = dict(Counter(labels.get(c.uid, "unlabeled") for c in ranked))
        out["precision_at"] = {}
        for k in (10, 25, 50, 100):
            p, n = precision_at_k(ranked, labels, k)
            out["precision_at"][k] = {"p": p, "decisive": n}
        out["average_precision"] = average_precision(ranked, labels)
    if hits is not None:
        out["hit_rate_at"] = {k: hit_rate_at_k((c.uid for c in ranked), hits, k) for k in (10, 25, 50, 100)}
        out["n_hits"] = len(hits)
    return out
