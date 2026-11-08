"""collections, collection_scores, screen_usage, probe_queue.collection_id

Revision ID: 002
Revises: 001
Create Date: 2026-11-08
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, Sequence[str], None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("models", sa.Column("embedding", sa.JSON(), nullable=True))

    op.create_table(
        "collections",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("domain_keywords", sa.JSON(), nullable=False),
        sa.Column("domain_tags", sa.JSON(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("tier", sa.Text(), nullable=False, server_default="standard"),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "collection_scores",
        sa.Column("collection_id", sa.Text(), primary_key=True),
        sa.Column("uid", sa.Text(), primary_key=True),
        sa.Column("ranker_version", sa.Text(), primary_key=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_collection_scores_score",
        "collection_scores",
        ["collection_id", "ranker_version", "score"],
    )
    op.create_table(
        "screen_usage",
        sa.Column("collection_id", sa.Text(), primary_key=True),
        sa.Column("year_month", sa.Text(), primary_key=True),
        sa.Column("enqueued_count", sa.Integer(), nullable=False, server_default="0"),
    )

    # Rebuild probe_queue with composite PK (collection_id, uid).
    # Drop indexes first — SQLite keeps index names across RENAME TABLE.
    op.drop_index("ix_probe_queue_status_priority", table_name="probe_queue")
    op.rename_table("probe_queue", "probe_queue_old")
    op.create_table(
        "probe_queue",
        sa.Column("collection_id", sa.Text(), primary_key=True),
        sa.Column("uid", sa.Text(), primary_key=True),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("claimed_by", sa.Text()),
    )
    op.create_index("ix_probe_queue_status_priority", "probe_queue", ["status", "priority"])
    op.create_index(
        "ix_probe_queue_collection_status",
        "probe_queue",
        ["collection_id", "status", "priority"],
    )
    op.execute(
        """
        INSERT INTO probe_queue
          (collection_id, uid, tier, priority, status, enqueued_at, claimed_at, claimed_by)
        SELECT '_legacy', uid, tier, priority, status, enqueued_at, claimed_at, claimed_by
        FROM probe_queue_old
        """
    )
    op.drop_table("probe_queue_old")


def downgrade() -> None:
    op.rename_table("probe_queue", "probe_queue_new")
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
    op.execute(
        """
        INSERT INTO probe_queue (uid, tier, priority, status, enqueued_at, claimed_at, claimed_by)
        SELECT DISTINCT ON (uid) uid, tier, priority, status, enqueued_at, claimed_at, claimed_by
        FROM probe_queue_new
        ORDER BY uid, priority DESC
        """
    )
    op.drop_table("probe_queue_new")
    op.drop_table("screen_usage")
    op.drop_index("ix_collection_scores_score", table_name="collection_scores")
    op.drop_table("collection_scores")
    op.drop_table("collections")
    op.drop_column("models", "embedding")
