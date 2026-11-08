"""CLI: kondor-crawl <subcommand>"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import kondor_crawler.sources  # noqa: F401
from kondor_crawler.collection import Collection
from kondor_crawler.config import Settings
from kondor_crawler.log import setup_logging
from kondor_crawler.rank import RANKER_VERSION, TIER_LIMITS
from kondor_crawler.source import SOURCES

_DEFAULT_SOURCES = ["civitai", "huggingface"]


def _store(settings: Settings):
    from kondor_crawler.db.store import Store

    if not settings.database_url:
        raise SystemExit("DATABASE_URL is required")
    return Store(settings.database_url)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _require_collection(store, collection_id: str | None) -> list[str] | None:
    if not collection_id:
        return None
    if store.get_collection(collection_id) is None:
        raise SystemExit(f"unknown collection {collection_id!r}")
    return [collection_id]


def cmd_migrate(_args, settings: Settings) -> int:
    store = _store(settings)
    store.migrate()
    print("schema migrated to head", file=sys.stderr)
    return 0


def cmd_crawl(args, settings: Settings) -> int:
    from kondor_crawler.jobs import run_job

    store = _store(settings)
    store.create_all()  # ponytail: upgrades use `migrate`; crawl just ensures tables exist
    collection_ids = _require_collection(store, args.collection)
    totals = run_job(
        store,
        settings,
        sources=args.sources,
        backfill=args.backfill,
        limit_per_source=args.limit_per_source,
        tier=args.tier,
        skip_enqueue=args.skip_enqueue,
        collection_ids=collection_ids,
    )
    print(json.dumps(totals, default=str), file=sys.stderr)
    return 0 if totals.get("ok") else 1


def cmd_rescore(args, settings: Settings) -> int:
    from kondor_crawler.jobs import rescore, rescore_collections

    store = _store(settings)
    n = rescore(store)
    collection_ids = _require_collection(store, args.collection)
    per = rescore_collections(store, collection_ids=collection_ids)
    print(
        f"scored {n} models globally with ranker {RANKER_VERSION}; "
        f"collections={per or '{}'}",
        file=sys.stderr,
    )
    return 0


def cmd_enqueue(args, settings: Settings) -> int:
    from kondor_crawler.jobs import enqueue_collections

    store = _store(settings)
    collection_ids = _require_collection(store, args.collection)
    per = enqueue_collections(store, collection_ids=collection_ids, tier=args.tier)
    if not per:
        print(
            "enqueued 0 (no collections registered — "
            "run: kondor-crawl collection upsert …)",
            file=sys.stderr,
        )
        return 0
    print(f"enqueued {sum(per.values())} by_collection={per}", file=sys.stderr)
    return 0


def cmd_collection(args, settings: Settings) -> int:
    store = _store(settings)
    store.create_all()
    if args.collection_cmd == "list":
        rows = [
            {
                "id": c.id,
                "name": c.name,
                "tier": c.tier,
                "keywords": c.domain_keywords,
                "tags": c.domain_tags,
            }
            for c in store.list_collections()
        ]
        print(json.dumps(rows, indent=2))
        return 0

    if args.collection_cmd == "upsert":
        keywords = _split_csv(args.keywords)
        tags = _split_csv(args.tags)
        description = args.description or ""
        if not keywords and not tags and not description.strip():
            print(
                "warning: collection has no domain keywords/tags/description; "
                "ranking will demote almost everything as domain_miss",
                file=sys.stderr,
            )
        col = Collection(
            id=args.id,
            name=args.name or args.id,
            domain_keywords=keywords,
            domain_tags=tags,
            description=description,
            tier=args.tier,
        )
        store.upsert_collection(col)
        print(json.dumps(col.to_dict(), indent=2, default=str))
        return 0

    raise SystemExit(f"unknown collection command {args.collection_cmd!r}")


def cmd_eval(args, settings: Settings) -> int:
    from kondor_crawler.eval import summarize
    from kondor_crawler.pipeline import read_jsonl
    from kondor_crawler.rank import rank_candidates

    labels: dict[str, str] = {}
    if args.labels:
        for line in Path(args.labels).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            labels[row["uid"]] = row["label"]

    if args.input:
        ranked = rank_candidates(read_jsonl(args.input))
        hits = None
    else:
        store = _store(settings)
        if args.collection:
            ranked = store.ranked_for(args.collection, RANKER_VERSION)
        else:
            ranked = store.ranked(RANKER_VERSION)
        hits = set(store.hit_uids()) or None

    report = summarize(ranked, labels=labels or None, hits=hits)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_export(args, settings: Settings) -> int:
    from kondor_crawler.pipeline import write_jsonl

    store = _store(settings)
    if args.collection:
        _require_collection(store, args.collection)
        ranked = store.ranked_for(args.collection, RANKER_VERSION)
    else:
        ranked = store.ranked(RANKER_VERSION)
    n = write_jsonl(args.out, ranked)
    print(f"wrote {n} → {args.out}", file=sys.stderr)
    return 0


def cmd_health(_args, settings: Settings) -> int:
    store = _store(settings)
    report = store.health()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 1


def cmd_claim(args, settings: Settings) -> int:
    store = _store(settings)
    if args.collection:
        _require_collection(store, args.collection)
    print(json.dumps(store.claim(args.worker, n=args.n, collection_id=args.collection)))
    return 0


def cmd_result(args, settings: Settings) -> int:
    store = _store(settings)
    detail = json.loads(args.detail) if args.detail else {}
    store.record_result(args.uid, args.result, detail)
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    settings = Settings.from_env()

    p = argparse.ArgumentParser(description="Kondor crawler")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument(
            "--sources",
            nargs="+",
            choices=sorted(SOURCES),
            default=list(_DEFAULT_SOURCES),
            help="default: civitai huggingface (pass github explicitly)",
        )
        sp.add_argument("--limit-per-source", type=int, default=None)
        sp.add_argument("--tier", choices=sorted(TIER_LIMITS), default=settings.default_tier)
        sp.add_argument("--backfill", action="store_true")
        sp.add_argument("--skip-enqueue", action="store_true")
        sp.add_argument(
            "--collection",
            default=None,
            help="limit rescore/enqueue to one collection id (default: all)",
        )
        return sp

    add_common(sub.add_parser("crawl", help="Crawl sources into Postgres"))
    add_common(sub.add_parser("run", help="Alias of crawl (cron entrypoint)"))

    rs = sub.add_parser("rescore", help="Re-score stored models (global + collections)")
    rs.add_argument("--collection", default=None)

    enq = sub.add_parser("enqueue", help="Build probe_queue from collection scores")
    enq.add_argument("--tier", choices=sorted(TIER_LIMITS), default=None)
    enq.add_argument("--collection", default=None)

    col = sub.add_parser("collection", help="Manage client collections")
    col_sub = col.add_subparsers(dest="collection_cmd", required=True)
    col_sub.add_parser("list", help="List collections")
    cup = col_sub.add_parser("upsert", help="Create or update a collection")
    cup.add_argument("--id", required=True)
    cup.add_argument("--name", default=None)
    cup.add_argument("--keywords", default="", help="comma-separated domain keywords")
    cup.add_argument("--tags", default="", help="comma-separated domain tags")
    cup.add_argument("--description", default="")
    cup.add_argument("--tier", choices=sorted(TIER_LIMITS), default="standard")

    ev = sub.add_parser("eval", help="Print ranking metrics")
    ev.add_argument("--labels", help="JSONL {uid,label} for proxy Precision@K")
    ev.add_argument("--input", help="JSONL candidates instead of DB")
    ev.add_argument("--collection", default=None)
    ex = sub.add_parser("export", help="Write ranked JSONL")
    ex.add_argument("-o", "--out", default="out/candidates.jsonl")
    ex.add_argument("--collection", default=None)
    sub.add_parser("migrate", help="Create tables")
    sub.add_parser("health", help="Exit 1 if stale / failing")
    cl = sub.add_parser("claim", help="Claim probe_queue rows (for probe workers)")
    cl.add_argument("--worker", default="cli")
    cl.add_argument("-n", type=int, default=1)
    cl.add_argument("--collection", default=None)
    rslt = sub.add_parser("result", help="Record a probe result")
    rslt.add_argument("uid")
    rslt.add_argument("result", choices=["hit", "miss", "error"])
    rslt.add_argument("--detail", default="")

    args = p.parse_args(argv)
    handlers = {
        "crawl": cmd_crawl,
        "run": cmd_crawl,
        "rescore": cmd_rescore,
        "enqueue": cmd_enqueue,
        "collection": cmd_collection,
        "eval": cmd_eval,
        "export": cmd_export,
        "migrate": cmd_migrate,
        "health": cmd_health,
        "claim": cmd_claim,
        "result": cmd_result,
    }
    return handlers[args.cmd](args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
