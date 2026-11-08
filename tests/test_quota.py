from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.collection import Collection
from kondor_crawler.jobs import enqueue_collections, rescore_collections
from kondor_crawler.queue import enqueue_tier
from kondor_crawler.rank import RANKER_VERSION, TIER_LIMITS


def _models(n: int) -> list[ModelCandidate]:
    out = []
    for i in range(n):
        out.append(
            ModelCandidate(
                source="civitai",
                source_id=str(i),
                name=f"product-lora-{i}",
                url="u",
                kind="lora",
                description="product photography studio commercial bottle skincare trained on my photos",
                tags=["commercial", "product"],
                trigger_words=[f"sks{i}"],
            )
        )
    return out


def test_quota_caps_enqueue(store):
    store.upsert_collection(
        Collection(
            id="pilot",
            name="Pilot",
            domain_keywords=["product photography", "skincare", "bottle"],
            domain_tags=["commercial"],
            description="commercial product photography",
            tier="free",  # free = 25
        )
    )
    store.upsert_models(_models(40))
    rescore_collections(store)
    ranked = store.ranked_for("pilot", RANKER_VERSION)
    n = enqueue_tier(store, ranked, "free", collection_id="pilot", quota_limit=25)
    assert n == TIER_LIMITS["free"]
    assert store.screen_usage_count("pilot") == n
    assert store.remaining_quota("pilot") == 0

    # Second pass: quota exhausted → 0 new enqueues
    n2 = enqueue_tier(store, ranked, "free", collection_id="pilot", quota_limit=0)
    assert n2 == 0
    assert store.screen_usage_count("pilot") == TIER_LIMITS["free"]


def test_enqueue_collections_respects_monthly_quota(store):
    store.upsert_collection(
        Collection(
            id="std",
            name="Std",
            domain_keywords=["product photography"],
            domain_tags=["commercial"],
            description="product",
            tier="free",
        )
    )
    store.upsert_models(_models(30))
    rescore_collections(store)
    first = enqueue_collections(store)
    assert first["std"] == TIER_LIMITS["free"]
    second = enqueue_collections(store)
    assert second["std"] == 0


def test_enterprise_uncapped(store):
    store.upsert_collection(
        Collection(
            id="ent",
            name="Ent",
            domain_keywords=["product photography"],
            domain_tags=["commercial"],
            description="product",
            tier="enterprise",
        )
    )
    assert store.remaining_quota("ent") is None


def test_enqueue_fills_past_already_probed_topn(store):
    """Quota must keep filling from lower ranks when the head is already probed."""
    store.upsert_collection(
        Collection(
            id="fill",
            name="Fill",
            domain_keywords=["product photography"],
            domain_tags=["commercial"],
            description="product",
            tier="free",
        )
    )
    store.upsert_models(_models(30))
    rescore_collections(store)
    ranked = store.ranked_for("fill", RANKER_VERSION)
    # Probe the entire free-tier-sized head (25) so a blind top-N slice would stall.
    for c in ranked[:25]:
        store.record_result(c.uid, "miss")
    assert store.remaining_quota("fill") == TIER_LIMITS["free"]
    n = enqueue_tier(
        store,
        ranked,
        "free",
        collection_id="fill",
        quota_limit=store.remaining_quota("fill"),
    )
    assert n == 5  # models 26–30 only
    assert store.screen_usage_count("fill") == 5
