"""init crawler tables

Revision ID: 001
Revises:
Create Date: 2026-11-08
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("uid", sa.Text(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=False, server_default=""),
        sa.Column("kind", sa.Text(), nullable=False, server_default=""),
        sa.Column("base_model", sa.Text(), nullable=False, server_default=""),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("trigger_words", sa.JSON(), nullable=False),
        sa.Column("claimed_datasets", sa.JSON(), nullable=False),
        sa.Column("flags", sa.JSON(), nullable=False),
        sa.Column("downloads", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("likes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.Text()),
        sa.Column("updated_at", sa.Text()),
        sa.Column("preview_urls", sa.JSON(), nullable=False),
        sa.Column("raw", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_models_source", "models", ["source"])
    op.create_index("ix_models_author", "models", ["author"])
    op.create_table(
        "scores",
        sa.Column("uid", sa.Text(), primary_key=True),
        sa.Column("ranker_version", sa.Text(), primary_key=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_scores_score", "scores", ["ranker_version", "score"])
    op.create_table(
        "crawl_state",
        sa.Column("source", sa.Text(), primary_key=True),
        sa.Column("cursor", sa.Text()),
        sa.Column("last_success_at", sa.Text()),
        sa.Column("last_run_id", sa.Integer()),
    )
    op.create_table(
        "crawl_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("ok", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sources", sa.JSON(), nullable=False),
        sa.Column("fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("new_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text()),
        sa.Column("detail", sa.JSON()),
    )
    op.create_table(
        "probe_queue",
        sa.Column("uid", sa.Text(), primary_key=True),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("claimed_by", sa.Text()),
    )
    op.create_index("ix_probe_queue_status_priority", "probe_queue", ["status", "priority"])
    op.create_table(
        "probe_results",
        sa.Column("uid", sa.Text(), primary_key=True),
        sa.Column("result", sa.Text(), nullable=False),
        sa.Column("detail", sa.JSON()),
        sa.Column("probed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("probe_results")
    op.drop_table("probe_queue")
    op.drop_table("crawl_runs")
    op.drop_table("crawl_state")
    op.drop_table("scores")
    op.drop_table("models")
