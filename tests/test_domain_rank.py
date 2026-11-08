from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.collection import Collection
from kondor_crawler.rank import domain_relevance, rank_for_collection, score_candidate


def _collection(**kw):
    return Collection(
        id="acme-product",
        name="Acme Product Pack",
        domain_keywords=kw.get("keywords", ["product photography", "bottle", "skincare"]),
        domain_tags=kw.get("tags", ["commercial", "studio"]),
        description=kw.get(
            "description", "Commercial product photography of skincare bottles on white"
        ),
        tier="standard",
    )


def test_domain_overlap_boosts_relevant_lora():
    col = _collection()
    relevant = ModelCandidate(
        source="civitai",
        source_id="1",
        name="SkincareBottleLoRA",
        url="u",
        kind="lora",
        description="DreamBooth trained on my photos of skincare bottles, product photography studio",
        tags=["commercial", "product"],
        trigger_words=["sks bottle"],
    )
    unrelated = ModelCandidate(
        source="civitai",
        source_id="2",
        name="AnimeWaifu",
        url="u",
        kind="lora",
        description="anime character style, trained on my private dataset",
        tags=["anime"],
        trigger_words=["waifu xyz"],
    )
    ranked = rank_for_collection([relevant, unrelated], col)
    by = {c.uid: c for c in ranked}
    assert by["civitai:1"].score > by["civitai:2"].score
    assert any(r.startswith("domain_") for r in by["civitai:1"].score_reasons)
    assert "domain_miss" in by["civitai:2"].score_reasons or "domain_miss_strong_custom" in by[
        "civitai:2"
    ].score_reasons


def test_zero_overlap_demotes_without_strong_custom():
    col = _collection()
    weak = ModelCandidate(
        source="civitai",
        source_id="3",
        name="GenericStyle",
        url="u",
        kind="lora",
        description="style transfer vibes",
        tags=["style"],
        trigger_words=["abc style", "def look", "ghi tone"],  # triggers alone = mild custom
    )
    boost, reasons = domain_relevance(col, weak)
    assert boost == 0
    ranked = rank_for_collection([weak], col)
    assert "domain_miss" in ranked[0].score_reasons
    # Demotion applied relative to suspicion-only score
    base = score_candidate(
        ModelCandidate(
            source="civitai",
            source_id="3",
            name="GenericStyle",
            url="u",
            kind="lora",
            description="style transfer vibes",
            tags=["style"],
            trigger_words=["abc style", "def look", "ghi tone"],
        )
    )
    assert ranked[0].score < base.score


def test_strong_custom_keeps_slot_without_domain():
    col = _collection()
    strong = ModelCandidate(
        source="civitai",
        source_id="4",
        name="PrivatePackLeak",
        url="u",
        kind="lora",
        description="DreamBooth trained on my private dataset of 200 photos",
        tags=["person"],
        trigger_words=["sks person"],
    )
    ranked = rank_for_collection([strong], col)
    assert "domain_miss_strong_custom" in ranked[0].score_reasons
    assert "domain_miss" not in ranked[0].score_reasons
