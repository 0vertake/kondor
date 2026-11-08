"""SQLAlchemy table definitions."""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
)
from sqlalchemy.types import JSON

metadata = MetaData()

models = Table(
    "models",
    metadata,
    Column("uid", Text, primary_key=True),
    Column("source", Text, nullable=False),
    Column("source_id", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("url", Text, nullable=False),
    Column("author", Text, nullable=False, server_default=""),
    Column("kind", Text, nullable=False, server_default=""),
    Column("base_model", Text, nullable=False, server_default=""),
    Column("tags", JSON, nullable=False),
    Column("description", Text, nullable=False, server_default=""),
    Column("trigger_words", JSON, nullable=False),
    Column("claimed_datasets", JSON, nullable=False),
    Column("flags", JSON, nullable=False),
    Column("downloads", Integer, nullable=False, server_default="0"),
    Column("likes", Integer, nullable=False, server_default="0"),
    Column("created_at", Text),
    Column("updated_at", Text),
    Column("preview_urls", JSON, nullable=False),
    Column("raw", JSON, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("first_seen", DateTime(timezone=True), nullable=False),
    Column("last_seen", DateTime(timezone=True), nullable=False),
    # Reserved for a later embedder; unused in scoring today.
    Column("embedding", JSON, nullable=True),
    Index("ix_models_source", "source"),
    Index("ix_models_author", "author"),
)

scores = Table(
    "scores",
    metadata,
    Column("uid", Text, primary_key=True),
    Column("ranker_version", Text, primary_key=True),
    Column("score", Float, nullable=False),
    Column("reasons", JSON, nullable=False),
    Column("scored_at", DateTime(timezone=True), nullable=False),
    Index("ix_scores_score", "ranker_version", "score"),
)

collections = Table(
    "collections",
    metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("domain_keywords", JSON, nullable=False),
    Column("domain_tags", JSON, nullable=False),
    Column("description", Text, nullable=False, server_default=""),
    Column("tier", Text, nullable=False, server_default="standard"),
    # Reserved for a later embedder; unused in scoring today.
    Column("embedding", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

collection_scores = Table(
    "collection_scores",
    metadata,
    Column("collection_id", Text, primary_key=True),
    Column("uid", Text, primary_key=True),
    Column("ranker_version", Text, primary_key=True),
    Column("score", Float, nullable=False),
    Column("reasons", JSON, nullable=False),
    Column("scored_at", DateTime(timezone=True), nullable=False),
    Index("ix_collection_scores_score", "collection_id", "ranker_version", "score"),
)

screen_usage = Table(
    "screen_usage",
    metadata,
    Column("collection_id", Text, primary_key=True),
    Column("year_month", Text, primary_key=True),  # YYYY-MM
    Column("enqueued_count", Integer, nullable=False, server_default="0"),
)

crawl_state = Table(
    "crawl_state",
    metadata,
    Column("source", Text, primary_key=True),
    Column("cursor", Text),
    Column("last_success_at", Text),
    Column("last_run_id", Integer),
)

crawl_runs = Table(
    "crawl_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("ok", Integer, nullable=False, server_default="0"),
    Column("sources", JSON, nullable=False),
    Column("fetched", Integer, nullable=False, server_default="0"),
    Column("new_count", Integer, nullable=False, server_default="0"),
    Column("updated_count", Integer, nullable=False, server_default="0"),
    Column("failed", Integer, nullable=False, server_default="0"),
    Column("error", Text),
    Column("detail", JSON),
)

probe_queue = Table(
    "probe_queue",
    metadata,
    Column("collection_id", Text, primary_key=True),
    Column("uid", Text, primary_key=True),
    Column("tier", Text, nullable=False),
    Column("priority", Float, nullable=False),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("enqueued_at", DateTime(timezone=True), nullable=False),
    Column("claimed_at", DateTime(timezone=True)),
    Column("claimed_by", Text),
    Index("ix_probe_queue_status_priority", "status", "priority"),
    Index("ix_probe_queue_collection_status", "collection_id", "status", "priority"),
)

probe_results = Table(
    "probe_results",
    metadata,
    Column("uid", Text, primary_key=True),
    Column("result", Text, nullable=False),
    Column("detail", JSON),
    Column("probed_at", DateTime(timezone=True), nullable=False),
)
