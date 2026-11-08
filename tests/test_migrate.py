from pathlib import Path

from sqlalchemy import inspect

from kondor_crawler.db.store import Store


def test_alembic_migrate_creates_collection_tables(tmp_path):
    db = tmp_path / "mig.db"
    url = f"sqlite:///{db}"
    store = Store(url)
    store.migrate()
    insp = inspect(store.engine)
    tables = set(insp.get_table_names())
    assert "collections" in tables
    assert "collection_scores" in tables
    assert "screen_usage" in tables
    assert "probe_queue" in tables
    assert "alembic_version" in tables
    cols = {c["name"] for c in insp.get_columns("probe_queue")}
    assert "collection_id" in cols
    assert "uid" in cols
    model_cols = {c["name"] for c in insp.get_columns("models")}
    assert "embedding" in model_cols
    # Idempotent
    store.migrate()
    assert Path(db).exists()
