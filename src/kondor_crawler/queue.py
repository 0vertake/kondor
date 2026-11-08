"""Tiered probe queue — the crawler/probe team boundary."""

from __future__ import annotations

from sqlalchemy import select

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.db import schema as s
from kondor_crawler.db.store import Store
from kondor_crawler.rank import TIER_LIMITS


def enqueue_tier(
    store: Store,
    ranked: list[ModelCandidate],
    tier: str,
    *,
    collection_id: str,
    quota_limit: int | None = None,
) -> int:
    """Fill probe_queue from ranked list until budget; skip unchanged already-probed."""
    if tier not in TIER_LIMITS:
        raise KeyError(f"Unknown tier {tier!r}. Available: {sorted(TIER_LIMITS)}")
    # ponytail: budget is remaining monthly quota (or tier cap); walk past probed head
    budget = quota_limit if quota_limit is not None else TIER_LIMITS[tier]

    probed = store.probed_map()
    with store.connect() as conn:
        last_seen = {
            r.uid: r.last_seen
            for r in conn.execute(select(s.models.c.uid, s.models.c.last_seen)).all()
        }
    items: list[tuple[str, str, str, float]] = []
    for c in ranked:
        if budget is not None and len(items) >= budget:
            break
        last = probed.get(c.uid)
        seen = last_seen.get(c.uid)
        if last is None or (seen is not None and seen > last):
            items.append((collection_id, c.uid, tier, c.score))
    n = store.enqueue(items)
    if n:
        store.record_screen_usage(collection_id, n)
    return n
