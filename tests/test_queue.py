from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.collection import Collection
from kondor_crawler.db import schema as s
from kondor_crawler.queue import enqueue_tier
from kondor_crawler.rank import RANKER_VERSION, rank_for_collection

COLL = "demo"


def _seed_collection(store, tier="free"):
    store.upsert_collection(
        Collection(
            id=COLL,
            name="Demo",
            domain_keywords=["product photography", "studio"],
            domain_tags=["commercial"],
            description="commercial product photo packs",
            tier=tier,
        )
    )


def _c(i, score=1.0, **kw):
    c = ModelCandidate(
        source="civitai",
        source_id=str(i),
        name=str(i),
        url="u",
        kind="lora",
        description=kw.pop("description", "product photography studio commercial"),
        **kw,
    )
    c.score = score
    return c


def test_enqueue_skips_already_probed(store):
    _seed_collection(store)
    cands = [_c(1, 9), _c(2, 8), _c(3, 7)]
    store.upsert_models(cands)
    ranked = rank_for_collection(cands, store.get_collection(COLL))
    store.save_collection_scores(COLL, ranked, RANKER_VERSION)
    ranked = store.ranked_for(COLL, RANKER_VERSION)
    n = enqueue_tier(store, ranked, "free", collection_id=COLL)
    assert n >= 3
    store.record_result("civitai:1", "miss")
    enqueue_tier(store, ranked, "free", collection_id=COLL)
    pending = store.claim("w", n=10, collection_id=COLL)
    assert all(item["uid"] != "civitai:1" for item in pending)


def test_claim_does_not_double_give(store):
    _seed_collection(store)
    cands = [_c(1, 5), _c(2, 4)]
    store.upsert_models(cands)
    store.enqueue([(COLL, "civitai:1", "free", 5), (COLL, "civitai:2", "free", 4)])
    first = store.claim("a", n=1, collection_id=COLL)
    second = store.claim("b", n=2, collection_id=COLL)
    assert len(first) == 1
    assert first[0]["uid"] not in {x["uid"] for x in second}
    assert len({x["uid"] for x in first + second}) == 2
    assert all("collection_id" in x and "uid" in x for x in first + second)


def test_claim_orders_by_priority_then_uid(store):
    _seed_collection(store)
    store.upsert_models([_c(1), _c(2), _c(3)])
    # Same priority → uid ASC tiebreak; higher priority first
    store.enqueue(
        [
            (COLL, "civitai:2", "free", 5.0),
            (COLL, "civitai:1", "free", 5.0),
            (COLL, "civitai:3", "free", 9.0),
        ]
    )
    claimed = store.claim("w", n=3, collection_id=COLL)
    assert [x["uid"] for x in claimed] == ["civitai:3", "civitai:1", "civitai:2"]


def test_requeue_if_changed_after_probe(store):
    _seed_collection(store)
    c = _c(1, 5, description="product photography studio v1")
    store.upsert_models([c])
    ranked = rank_for_collection([c], store.get_collection(COLL))
    store.save_collection_scores(COLL, ranked, RANKER_VERSION)
    ranked = store.ranked_for(COLL, RANKER_VERSION)
    enqueue_tier(store, ranked, "free", collection_id=COLL)
    store.record_result("civitai:1", "miss")
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    with store.connect() as conn:
        conn.execute(update(s.models).where(s.models.c.uid == "civitai:1").values(last_seen=future))
    n = enqueue_tier(store, ranked, "free", collection_id=COLL)
    assert n == 1
    claimed = store.claim("w", n=1, collection_id=COLL)
    assert claimed == [{"collection_id": COLL, "uid": "civitai:1"}]
