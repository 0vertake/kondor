"""Postgres (or SQLite) persistence."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.collection import Collection
from kondor_crawler.db import schema as s
from kondor_crawler.rank import TIER_LIMITS

_LOCK_KEY = 0x4B4F4E44  # 'KOND'


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, url: str):
        if not url:
            raise ValueError("DATABASE_URL is required")
        kwargs: dict[str, Any] = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        self.url = url
        self.engine: Engine = create_engine(url, **kwargs)
        self.dialect = self.engine.dialect.name

    def create_all(self) -> None:
        s.metadata.create_all(self.engine)

    def migrate(self) -> None:
        """Apply Alembic migrations to head (prod upgrade path)."""
        from pathlib import Path

        from alembic import command
        from alembic.config import Config

        root = Path(__file__).resolve().parents[3]
        candidates = [Path("alembic.ini"), root / "alembic.ini"]
        ini = next((p for p in candidates if p.is_file()), None)
        if ini is None:
            raise FileNotFoundError("alembic.ini not found; cannot migrate")
        cfg = Config(str(ini))
        # ConfigParser interpolation treats % specially
        cfg.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
        script_candidates = [Path("alembic"), root / "alembic", ini.parent / "alembic"]
        script_loc = next((p for p in script_candidates if p.is_dir()), None)
        if script_loc is not None:
            cfg.set_main_option("script_location", str(script_loc))
        command.upgrade(cfg, "head")

    @contextmanager
    def connect(self):
        with self.engine.begin() as conn:
            yield conn

    @contextmanager
    def advisory_lock(self):
        # Hold one connection for the whole critical section — PG session locks
        # are released when the connection closes.
        if self.dialect != "postgresql":
            yield
            return
        conn = self.engine.connect()
        conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _LOCK_KEY})
        try:
            yield
        finally:
            try:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _LOCK_KEY})
                conn.commit()
            finally:
                conn.close()

    # --- models ---

    def upsert_models(self, candidates: Iterable[ModelCandidate]) -> dict[str, int]:
        now = _now()
        new = updated = unchanged = 0
        with self.connect() as conn:
            for c in candidates:
                h = c.content_hash()
                row = conn.execute(
                    select(s.models.c.uid, s.models.c.content_hash).where(s.models.c.uid == c.uid)
                ).fetchone()
                values = {
                    "uid": c.uid,
                    "source": c.source,
                    "source_id": c.source_id,
                    "name": c.name,
                    "url": c.url,
                    "author": c.author,
                    "kind": c.kind,
                    "base_model": c.base_model,
                    "tags": list(c.tags),
                    "description": c.description,
                    "trigger_words": list(c.trigger_words),
                    "claimed_datasets": list(c.claimed_datasets),
                    "flags": list(c.flags),
                    "downloads": c.downloads,
                    "likes": c.likes,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                    "preview_urls": list(c.preview_urls),
                    "raw": dict(c.raw),
                    "content_hash": h,
                    "last_seen": now,
                }
                if row is None:
                    values["first_seen"] = now
                    conn.execute(s.models.insert().values(**values))
                    new += 1
                elif row.content_hash == h:
                    conn.execute(
                        s.models.update().where(s.models.c.uid == c.uid).values(last_seen=now)
                    )
                    unchanged += 1
                else:
                    conn.execute(s.models.update().where(s.models.c.uid == c.uid).values(**values))
                    updated += 1
        return {"new": new, "updated": updated, "unchanged": unchanged}

    def get_model(self, uid: str) -> ModelCandidate | None:
        with self.connect() as conn:
            row = conn.execute(select(s.models).where(s.models.c.uid == uid)).mappings().fetchone()
        return _row_to_candidate(row) if row else None

    def load_models(self, uids: Iterable[str] | None = None) -> list[ModelCandidate]:
        stmt = select(s.models)
        uids = list(uids) if uids is not None else None
        if uids is not None:
            if not uids:
                return []
            stmt = stmt.where(s.models.c.uid.in_(uids))
        with self.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [_row_to_candidate(r) for r in rows]

    # --- scores ---

    def save_scores(self, candidates: Iterable[ModelCandidate], ranker_version: str) -> int:
        now = _now()
        n = 0
        with self.connect() as conn:
            for c in candidates:
                existing = conn.execute(
                    select(s.scores.c.uid).where(
                        s.scores.c.uid == c.uid, s.scores.c.ranker_version == ranker_version
                    )
                ).fetchone()
                values = {
                    "uid": c.uid,
                    "ranker_version": ranker_version,
                    "score": c.score,
                    "reasons": list(c.score_reasons),
                    "scored_at": now,
                }
                if existing:
                    conn.execute(
                        s.scores.update()
                        .where(s.scores.c.uid == c.uid, s.scores.c.ranker_version == ranker_version)
                        .values(score=c.score, reasons=list(c.score_reasons), scored_at=now)
                    )
                else:
                    conn.execute(s.scores.insert().values(**values))
                n += 1
        return n

    def ranked(self, ranker_version: str) -> list[ModelCandidate]:
        with self.connect() as conn:
            rows = (
                conn.execute(
                    select(s.models, s.scores.c.score, s.scores.c.reasons)
                    .join(s.scores, s.models.c.uid == s.scores.c.uid)
                    .where(s.scores.c.ranker_version == ranker_version)
                    .order_by(s.scores.c.score.desc())
                )
                .mappings()
                .all()
            )
        out = []
        for row in rows:
            c = _row_to_candidate(row)
            c.score = float(row["score"] or 0)
            c.score_reasons = list(row["reasons"] or [])
            out.append(c)
        return out

    # --- collections ---

    def upsert_collection(self, collection: Collection) -> None:
        now = _now()
        with self.connect() as conn:
            existing = conn.execute(
                select(s.collections.c.id).where(s.collections.c.id == collection.id)
            ).fetchone()
            values = {
                "id": collection.id,
                "name": collection.name,
                "domain_keywords": list(collection.domain_keywords),
                "domain_tags": list(collection.domain_tags),
                "description": collection.description or "",
                "tier": collection.tier,
                "embedding": collection.embedding,
                "updated_at": now,
            }
            if existing:
                conn.execute(
                    s.collections.update()
                    .where(s.collections.c.id == collection.id)
                    .values(**values)
                )
            else:
                values["created_at"] = now
                conn.execute(s.collections.insert().values(**values))

    def get_collection(self, collection_id: str) -> Collection | None:
        with self.connect() as conn:
            row = (
                conn.execute(select(s.collections).where(s.collections.c.id == collection_id))
                .mappings()
                .fetchone()
            )
        return _row_to_collection(row) if row else None

    def list_collections(self) -> list[Collection]:
        with self.connect() as conn:
            rows = conn.execute(select(s.collections).order_by(s.collections.c.id)).mappings().all()
        return [_row_to_collection(r) for r in rows]

    def save_collection_scores(
        self,
        collection_id: str,
        candidates: Iterable[ModelCandidate],
        ranker_version: str,
    ) -> int:
        now = _now()
        n = 0
        with self.connect() as conn:
            for c in candidates:
                existing = conn.execute(
                    select(s.collection_scores.c.uid).where(
                        s.collection_scores.c.collection_id == collection_id,
                        s.collection_scores.c.uid == c.uid,
                        s.collection_scores.c.ranker_version == ranker_version,
                    )
                ).fetchone()
                values = {
                    "collection_id": collection_id,
                    "uid": c.uid,
                    "ranker_version": ranker_version,
                    "score": c.score,
                    "reasons": list(c.score_reasons),
                    "scored_at": now,
                }
                if existing:
                    conn.execute(
                        s.collection_scores.update()
                        .where(
                            s.collection_scores.c.collection_id == collection_id,
                            s.collection_scores.c.uid == c.uid,
                            s.collection_scores.c.ranker_version == ranker_version,
                        )
                        .values(score=c.score, reasons=list(c.score_reasons), scored_at=now)
                    )
                else:
                    conn.execute(s.collection_scores.insert().values(**values))
                n += 1
        return n

    def ranked_for(self, collection_id: str, ranker_version: str) -> list[ModelCandidate]:
        with self.connect() as conn:
            rows = (
                conn.execute(
                    select(s.models, s.collection_scores.c.score, s.collection_scores.c.reasons)
                    .join(
                        s.collection_scores,
                        s.models.c.uid == s.collection_scores.c.uid,
                    )
                    .where(
                        s.collection_scores.c.collection_id == collection_id,
                        s.collection_scores.c.ranker_version == ranker_version,
                    )
                    .order_by(s.collection_scores.c.score.desc(), s.models.c.uid.asc())
                )
                .mappings()
                .all()
            )
        out = []
        for row in rows:
            c = _row_to_candidate(row)
            c.score = float(row["score"] or 0)
            c.score_reasons = list(row["reasons"] or [])
            out.append(c)
        return out

    def screen_usage_count(self, collection_id: str, year_month: str | None = None) -> int:
        ym = year_month or _year_month()
        with self.connect() as conn:
            row = conn.execute(
                select(s.screen_usage.c.enqueued_count).where(
                    s.screen_usage.c.collection_id == collection_id,
                    s.screen_usage.c.year_month == ym,
                )
            ).fetchone()
        return int(row[0]) if row else 0

    def remaining_quota(
        self,
        collection_id: str,
        *,
        tier: str | None = None,
        year_month: str | None = None,
    ) -> int | None:
        """None means uncapped (enterprise)."""
        col = self.get_collection(collection_id)
        if col is None:
            raise KeyError(f"Unknown collection {collection_id!r}")
        t = tier or col.tier
        limit = TIER_LIMITS.get(t)
        if limit is None:
            return None
        used = self.screen_usage_count(collection_id, year_month)
        return max(0, limit - used)

    def record_screen_usage(
        self,
        collection_id: str,
        n: int,
        *,
        year_month: str | None = None,
    ) -> int:
        if n <= 0:
            return self.screen_usage_count(collection_id, year_month)
        ym = year_month or _year_month()
        with self.connect() as conn:
            row = conn.execute(
                select(s.screen_usage.c.enqueued_count).where(
                    s.screen_usage.c.collection_id == collection_id,
                    s.screen_usage.c.year_month == ym,
                )
            ).fetchone()
            if row:
                new_count = int(row[0]) + n
                conn.execute(
                    s.screen_usage.update()
                    .where(
                        s.screen_usage.c.collection_id == collection_id,
                        s.screen_usage.c.year_month == ym,
                    )
                    .values(enqueued_count=new_count)
                )
            else:
                new_count = n
                conn.execute(
                    s.screen_usage.insert().values(
                        collection_id=collection_id,
                        year_month=ym,
                        enqueued_count=new_count,
                    )
                )
        return new_count

    # --- crawl state / runs ---

    def get_crawl_state(self, source: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(select(s.crawl_state).where(s.crawl_state.c.source == source)).mappings().fetchone()
        if not row:
            return {"source": source, "cursor": None, "last_success_at": None, "last_run_id": None}
        return dict(row)

    def set_cursor(self, source: str, cursor: str | None, *, run_id: int | None = None) -> None:
        self._upsert_crawl_state(source, cursor=cursor, last_run_id=run_id)

    def mark_source_success(self, source: str, watermark: str, run_id: int) -> None:
        self._upsert_crawl_state(
            source, cursor=None, last_success_at=watermark, last_run_id=run_id
        )

    def _upsert_crawl_state(self, source: str, **values: Any) -> None:
        with self.connect() as conn:
            existing = conn.execute(
                select(s.crawl_state.c.source).where(s.crawl_state.c.source == source)
            ).fetchone()
            if existing:
                conn.execute(
                    s.crawl_state.update().where(s.crawl_state.c.source == source).values(**values)
                )
            else:
                conn.execute(s.crawl_state.insert().values(source=source, **values))

    def start_run(self, sources: list[str]) -> int:
        with self.connect() as conn:
            result = conn.execute(
                s.crawl_runs.insert().values(
                    started_at=_now(),
                    ok=0,
                    sources=list(sources),
                    fetched=0,
                    new_count=0,
                    updated_count=0,
                    failed=0,
                    detail={},
                )
            )
            return int(result.inserted_primary_key[0])

    def finish_run(
        self,
        run_id: int,
        *,
        ok: bool,
        fetched: int = 0,
        new_count: int = 0,
        updated_count: int = 0,
        failed: int = 0,
        error: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                s.crawl_runs.update()
                .where(s.crawl_runs.c.id == run_id)
                .values(
                    finished_at=_now(),
                    ok=1 if ok else 0,
                    fetched=fetched,
                    new_count=new_count,
                    updated_count=updated_count,
                    failed=failed,
                    error=error,
                    detail=detail or {},
                )
            )

    def latest_run(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(select(s.crawl_runs).order_by(s.crawl_runs.c.id.desc()).limit(1)).mappings().fetchone()
        return dict(row) if row else None

    def last_successful_run(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = (
                conn.execute(
                    select(s.crawl_runs)
                    .where(s.crawl_runs.c.ok == 1)
                    .order_by(s.crawl_runs.c.id.desc())
                    .limit(1)
                )
                .mappings()
                .fetchone()
            )
        return dict(row) if row else None

    # --- queue / results ---

    def probed_map(self) -> dict[str, datetime]:
        with self.connect() as conn:
            rows = conn.execute(select(s.probe_results.c.uid, s.probe_results.c.probed_at)).all()
        return {r.uid: r.probed_at for r in rows}

    def hit_uids(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                select(s.probe_results.c.uid).where(s.probe_results.c.result == "hit")
            ).all()
        return [r.uid for r in rows]

    def record_result(self, uid: str, result: str, detail: dict[str, Any] | None = None) -> None:
        now = _now()
        with self.connect() as conn:
            existing = conn.execute(select(s.probe_results.c.uid).where(s.probe_results.c.uid == uid)).fetchone()
            if existing:
                conn.execute(
                    s.probe_results.update()
                    .where(s.probe_results.c.uid == uid)
                    .values(result=result, detail=detail or {}, probed_at=now)
                )
            else:
                conn.execute(
                    s.probe_results.insert().values(
                        uid=uid, result=result, detail=detail or {}, probed_at=now
                    )
                )
            conn.execute(
                s.probe_queue.update()
                .where(s.probe_queue.c.uid == uid)
                .values(status="done" if result != "error" else "failed")
            )

    def enqueue(self, items: list[tuple[str, str, str, float]]) -> int:
        """items: (collection_id, uid, tier, priority). Skip existing pending/claimed.

        Returns count of *new* enqueues (status flipped to pending or inserted),
        not priority-only updates of already-pending rows.
        """
        now = _now()
        n = 0
        with self.connect() as conn:
            for collection_id, uid, tier, priority in items:
                row = conn.execute(
                    select(s.probe_queue.c.uid, s.probe_queue.c.status).where(
                        s.probe_queue.c.collection_id == collection_id,
                        s.probe_queue.c.uid == uid,
                    )
                ).fetchone()
                if row and row.status in {"pending", "claimed"}:
                    conn.execute(
                        s.probe_queue.update()
                        .where(
                            s.probe_queue.c.collection_id == collection_id,
                            s.probe_queue.c.uid == uid,
                        )
                        .values(tier=tier, priority=priority)
                    )
                    continue
                if row:
                    conn.execute(
                        s.probe_queue.update()
                        .where(
                            s.probe_queue.c.collection_id == collection_id,
                            s.probe_queue.c.uid == uid,
                        )
                        .values(
                            tier=tier,
                            priority=priority,
                            status="pending",
                            enqueued_at=now,
                            claimed_at=None,
                            claimed_by=None,
                        )
                    )
                else:
                    conn.execute(
                        s.probe_queue.insert().values(
                            collection_id=collection_id,
                            uid=uid,
                            tier=tier,
                            priority=priority,
                            status="pending",
                            enqueued_at=now,
                        )
                    )
                n += 1
        return n

    def claim(
        self,
        worker_id: str,
        n: int = 1,
        *,
        collection_id: str | None = None,
    ) -> list[dict[str, str]]:
        """Claim pending work. Returns [{"collection_id", "uid"}, ...] by priority."""
        now = _now()
        with self.connect() as conn:
            if self.dialect == "postgresql":
                if collection_id:
                    rows = conn.execute(
                        text(
                            """
                            UPDATE probe_queue AS q
                            SET status = 'claimed', claimed_at = :now, claimed_by = :wid
                            FROM (
                              SELECT collection_id, uid FROM probe_queue
                              WHERE status = 'pending' AND collection_id = :cid
                              ORDER BY priority DESC, uid ASC
                              FOR UPDATE SKIP LOCKED
                              LIMIT :n
                            ) AS pick
                            WHERE q.collection_id = pick.collection_id AND q.uid = pick.uid
                            RETURNING q.collection_id, q.uid
                            """
                        ),
                        {"now": now, "wid": worker_id, "n": n, "cid": collection_id},
                    ).fetchall()
                else:
                    rows = conn.execute(
                        text(
                            """
                            UPDATE probe_queue AS q
                            SET status = 'claimed', claimed_at = :now, claimed_by = :wid
                            FROM (
                              SELECT collection_id, uid FROM probe_queue
                              WHERE status = 'pending'
                              ORDER BY priority DESC, uid ASC
                              FOR UPDATE SKIP LOCKED
                              LIMIT :n
                            ) AS pick
                            WHERE q.collection_id = pick.collection_id AND q.uid = pick.uid
                            RETURNING q.collection_id, q.uid
                            """
                        ),
                        {"now": now, "wid": worker_id, "n": n},
                    ).fetchall()
                return [{"collection_id": r[0], "uid": r[1]} for r in rows]
            stmt = (
                select(s.probe_queue.c.collection_id, s.probe_queue.c.uid)
                .where(s.probe_queue.c.status == "pending")
                .order_by(s.probe_queue.c.priority.desc(), s.probe_queue.c.uid.asc())
                .limit(n)
            )
            if collection_id:
                stmt = stmt.where(s.probe_queue.c.collection_id == collection_id)
            rows = conn.execute(stmt).all()
            out: list[dict[str, str]] = []
            for collection_id_row, uid in rows:
                conn.execute(
                    s.probe_queue.update()
                    .where(
                        s.probe_queue.c.collection_id == collection_id_row,
                        s.probe_queue.c.uid == uid,
                    )
                    .values(status="claimed", claimed_at=now, claimed_by=worker_id)
                )
                out.append({"collection_id": collection_id_row, "uid": uid})
            return out

    def queue_status_counts(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                select(s.probe_queue.c.status, func.count()).group_by(s.probe_queue.c.status)
            ).all()
        return {r[0]: int(r[1]) for r in rows}

    def health(self, *, stale_hours: int = 24) -> dict[str, Any]:
        last_ok = self.last_successful_run()
        latest = self.latest_run()
        stale = True
        if last_ok and last_ok.get("finished_at"):
            finished = last_ok["finished_at"]
            if finished.tzinfo is None:
                finished = finished.replace(tzinfo=timezone.utc)
            stale = _now() - finished > timedelta(hours=stale_hours)
        failed_recent = 0
        with self.connect() as conn:
            cutoff = _now() - timedelta(hours=stale_hours)
            failed_recent = int(
                conn.execute(
                    select(func.count())
                    .select_from(s.crawl_runs)
                    .where(s.crawl_runs.c.ok == 0, s.crawl_runs.c.started_at >= cutoff)
                ).scalar()
                or 0
            )
            total_recent = int(
                conn.execute(
                    select(func.count())
                    .select_from(s.crawl_runs)
                    .where(s.crawl_runs.c.started_at >= cutoff)
                ).scalar()
                or 0
            )
        error_rate = (failed_recent / total_recent) if total_recent else 0.0
        collections = self.list_collections()
        quotas = []
        for col in collections:
            rem = self.remaining_quota(col.id)
            quotas.append(
                {
                    "collection_id": col.id,
                    "tier": col.tier,
                    "used": self.screen_usage_count(col.id),
                    "remaining": rem,
                }
            )
        ok = not stale and error_rate < 0.5
        return {
            "ok": ok,
            "stale": stale,
            "error_rate": error_rate,
            "last_success": last_ok,
            "latest": latest,
            "queue": self.queue_status_counts(),
            "collections": len(collections),
            "quotas": quotas,
        }


def _year_month(when: datetime | None = None) -> str:
    dt = when or _now()
    return f"{dt.year:04d}-{dt.month:02d}"


def _row_to_collection(row: Any) -> Collection:
    emb = row.get("embedding")
    return Collection(
        id=row["id"],
        name=row["name"],
        domain_keywords=list(row["domain_keywords"] or []),
        domain_tags=list(row["domain_tags"] or []),
        description=row["description"] or "",
        tier=row["tier"] or "standard",
        embedding=list(emb) if emb is not None else None,
    )


def _row_to_candidate(row: Any) -> ModelCandidate:
    return ModelCandidate(
        source=row["source"],
        source_id=row["source_id"],
        name=row["name"],
        url=row["url"],
        author=row["author"] or "",
        kind=row["kind"] or "",
        base_model=row["base_model"] or "",
        tags=list(row["tags"] or []),
        description=row["description"] or "",
        trigger_words=list(row["trigger_words"] or []),
        claimed_datasets=list(row["claimed_datasets"] or []),
        flags=list(row["flags"] or []),
        downloads=int(row["downloads"] or 0),
        likes=int(row["likes"] or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        preview_urls=list(row["preview_urls"] or []),
        raw=dict(row["raw"] or {}),
    )
