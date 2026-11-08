"""Crawl / rescore / enqueue / eval jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import kondor_crawler.sources  # noqa: F401
from kondor_crawler.collection import Collection
from kondor_crawler.config import Settings
from kondor_crawler.db.store import Store
from kondor_crawler.enrich import enrich_candidate, should_enrich
from kondor_crawler.httputil import BudgetExceeded, configure_http
from kondor_crawler.log import get_logger
from kondor_crawler.queue import enqueue_tier
from kondor_crawler.rank import (
    RANKER_VERSION,
    promote_related,
    rank_candidates,
    rank_for_collection,
)
from kondor_crawler.source import get_source

log = get_logger("kondor_crawler.jobs")


def _collections(store: Store, collection_ids: Sequence[str] | None) -> list[Collection]:
    if collection_ids is None:
        return store.list_collections()
    out: list[Collection] = []
    for cid in collection_ids:
        col = store.get_collection(cid)
        if col is None:
            raise KeyError(f"Unknown collection {cid!r}")
        out.append(col)
    return out


def _source_kwargs(name: str, settings: Settings) -> dict:
    kwargs: dict = {"page_size": settings.page_size}
    if name == "huggingface" and settings.hf_token:
        kwargs["token"] = settings.hf_token
    if name == "civitai" and settings.civitai_token:
        kwargs["token"] = settings.civitai_token
    if name == "github" and settings.github_token:
        kwargs["token"] = settings.github_token
    return kwargs


def crawl_sources(
    store: Store,
    settings: Settings,
    sources: Sequence[str],
    *,
    backfill: bool = False,
    limit_per_source: int | None = None,
    run_id: int,
) -> dict:
    configure_http(min_interval=settings.min_interval, budget=settings.request_budget)
    watermark = datetime.now(timezone.utc).isoformat()
    totals = {"fetched": 0, "new": 0, "updated": 0, "unchanged": 0, "failed": 0, "per_source": {}}
    enrich_left = settings.enrich_budget

    for name in sources:
        state = store.get_crawl_state(name)
        since = None if backfill else state.get("last_success_at")
        cursor = state.get("cursor")
        src = get_source(name, **_source_kwargs(name, settings))
        batch = []
        source_stats = {"fetched": 0, "error": None}
        try:

            def on_progress(cur: str | None, _name=name) -> None:
                store.set_cursor(_name, cur, run_id=run_id)

            for cand in src.iter_candidates(
                limit=limit_per_source,
                since=since,
                start_cursor=cursor,
                on_progress=on_progress,
            ):
                if enrich_left > 0 and should_enrich(cand):
                    try:
                        cand = enrich_candidate(cand)
                        enrich_left -= 1
                    except BudgetExceeded:
                        enrich_left = 0
                batch.append(cand)
                if len(batch) >= 50:
                    stats = store.upsert_models(batch)
                    source_stats["fetched"] += len(batch)
                    for k in ("new", "updated", "unchanged"):
                        totals[k] += stats[k]
                    batch.clear()
            if batch:
                stats = store.upsert_models(batch)
                source_stats["fetched"] += len(batch)
                for k in ("new", "updated", "unchanged"):
                    totals[k] += stats[k]
            store.mark_source_success(name, watermark, run_id)
        except Exception as e:
            totals["failed"] += 1
            source_stats["error"] = str(e)
            log.exception("source failed", extra={"data": {"source": name}})
        totals["fetched"] += source_stats["fetched"]
        totals["per_source"][name] = source_stats
        log.info("source done", extra={"data": {"source": name, **source_stats}})
    return totals


def rescore(store: Store) -> int:
    """Global suspicion scores (backward-compatible / eval path)."""
    models = store.load_models()
    ranked = rank_candidates(models)
    hits = store.hit_uids()
    if hits:
        ranked = promote_related(ranked, hits)
    return store.save_scores(ranked, RANKER_VERSION)


def rescore_collections(
    store: Store,
    *,
    collection_ids: Sequence[str] | None = None,
) -> dict[str, int]:
    """Per-collection domain ranking → collection_scores."""
    models = store.load_models()
    hits = store.hit_uids()
    scored: dict[str, int] = {}
    for col in _collections(store, collection_ids):
        ranked = rank_for_collection(models, col)
        if hits:
            ranked = promote_related(ranked, hits)
        scored[col.id] = store.save_collection_scores(col.id, ranked, RANKER_VERSION)
        log.info(
            "collection rescored",
            extra={"data": {"collection_id": col.id, "scored": scored[col.id]}},
        )
    return scored


def enqueue_collections(
    store: Store,
    *,
    collection_ids: Sequence[str] | None = None,
    tier: str | None = None,
) -> dict[str, int]:
    collections = _collections(store, collection_ids)
    if not collections:
        log.warning("no collections registered; skip enqueue")
        return {}

    enqueued: dict[str, int] = {}
    for col in collections:
        use_tier = tier or col.tier
        budget = store.remaining_quota(col.id, tier=use_tier)
        if budget == 0:
            log.info(
                "quota exhausted",
                extra={"data": {"collection_id": col.id, "tier": use_tier}},
            )
            enqueued[col.id] = 0
            continue
        ranked = store.ranked_for(col.id, RANKER_VERSION)
        enqueued[col.id] = enqueue_tier(
            store,
            ranked,
            use_tier,
            collection_id=col.id,
            quota_limit=budget,
        )
    return enqueued


def run_job(
    store: Store,
    settings: Settings,
    *,
    sources: Sequence[str] | None = None,
    backfill: bool = False,
    limit_per_source: int | None = None,
    tier: str | None = None,
    skip_enqueue: bool = False,
    collection_ids: Sequence[str] | None = None,
) -> dict:
    # ponytail: default excludes github (needs token); pass explicitly when ready
    names = list(sources) if sources is not None else ["civitai", "huggingface"]
    with store.advisory_lock():
        run_id = store.start_run(names)
        try:
            totals = crawl_sources(
                store,
                settings,
                names,
                backfill=backfill,
                limit_per_source=limit_per_source,
                run_id=run_id,
            )
            n_global = rescore(store)
            per_collection = rescore_collections(store, collection_ids=collection_ids)
            enqueued: dict[str, int] = {}
            if not skip_enqueue:
                if not per_collection and collection_ids is None:
                    log.warning(
                        "no collections registered; crawl+global score done, enqueue skipped"
                    )
                else:
                    enqueued = enqueue_collections(
                        store, collection_ids=collection_ids, tier=tier
                    )
            ok = totals["failed"] == 0
            store.finish_run(
                run_id,
                ok=ok,
                fetched=totals["fetched"],
                new_count=totals["new"],
                updated_count=totals["updated"],
                failed=totals["failed"],
                error=None if ok else "one or more sources failed",
                detail={
                    "per_source": totals["per_source"],
                    "scored_global": n_global,
                    "scored_collections": per_collection,
                    "enqueued": enqueued,
                },
            )
            totals["scored"] = n_global
            totals["scored_collections"] = per_collection
            totals["enqueued"] = sum(enqueued.values())
            totals["enqueued_by_collection"] = enqueued
            totals["ok"] = ok
            totals["run_id"] = run_id
            log.info("run finished", extra={"data": totals})
            return totals
        except Exception as e:
            store.finish_run(run_id, ok=False, error=str(e))
            raise
