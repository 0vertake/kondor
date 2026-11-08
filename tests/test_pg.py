"""Postgres integration tests. Skipped unless TEST_DATABASE_URL is set."""

from __future__ import annotations

import os
import uuid

import pytest

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.collection import Collection
from kondor_crawler.db.store import Store
from kondor_crawler.rank import RANKER_VERSION

pytestmark = pytest.mark.pg


@pytest.fixture
def store():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    s = Store(url)
    s.create_all()
    return s


def test_pg_upsert_claim_and_lock(store: Store):
    uid_src = uuid.uuid4().hex[:8]
    coll = f"c-{uid_src}"
    store.upsert_collection(
        Collection(
            id=coll,
            name="pg",
            domain_keywords=["product"],
            domain_tags=["commercial"],
            tier="free",
        )
    )
    c = ModelCandidate(source="test", source_id=uid_src, name="n", url="u", kind="lora")
    store.upsert_models([c])
    store.save_scores([c], RANKER_VERSION)
    store.save_collection_scores(coll, [c], RANKER_VERSION)
    store.enqueue([(coll, c.uid, "free", 1.0)])
    claimed = store.claim("worker-a", n=1, collection_id=coll)
    assert claimed == [{"collection_id": coll, "uid": c.uid}]
    claimed2 = store.claim("worker-b", n=1, collection_id=coll)
    assert claimed2 == []
    with store.advisory_lock():
        pass
    h = store.health(stale_hours=24)
    assert "ok" in h

