from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.rank import apply_tier, promote_related, score_candidate


def test_custom_lora_beats_popular_photoreal_foundationish():
    custom = score_candidate(
        ModelCandidate(
            source="civitai",
            source_id="1",
            name="MyProduct",
            url="http://x",
            kind="lora",
            description="DreamBooth trained on my photos, private dataset",
            tags=["photorealistic", "product"],
            trigger_words=["sks product"],
        )
    )
    juggernaut = score_candidate(
        ModelCandidate(
            source="huggingface",
            source_id="RunDiffusion/Juggernaut-XL-v9",
            name="Juggernaut-XL-v9",
            url="http://x",
            kind="finetune",
            description="photorealistic realistic portrait studio raw photo",
            tags=["photorealistic", "text-to-image"],
            downloads=500_000,
            likes=10_000,
        )
    )
    foundation = score_candidate(
        ModelCandidate(
            source="huggingface",
            source_id="stabilityai/stable-diffusion-xl-base-1.0",
            name="sdxl",
            url="http://x",
            kind="finetune",
            tags=["diffusers", "text-to-image"],
            downloads=5_000_000,
            flags=["foundation"],
        )
    )
    merge = score_candidate(
        ModelCandidate(
            source="civitai",
            source_id="9",
            name="AnimeMerge",
            url="http://x",
            kind="checkpoint",
            description="Merged mix of baked anime checkpoints",
            tags=["anime", "merge"],
            flags=["merge"],
        )
    )
    assert custom.score > juggernaut.score
    assert custom.score > foundation.score
    assert custom.score > merge.score
    assert "foundation_demote" in foundation.score_reasons
    assert "mega_popular_demote" in juggernaut.score_reasons


def test_photo_hints_alone_do_not_beat_custom_lora():
    photo_only = score_candidate(
        ModelCandidate(
            source="huggingface",
            source_id="x/realistic-vision",
            name="rv",
            url="u",
            kind="finetune",
            description="photorealistic realistic portrait studio commercial product",
            downloads=50_000,
        )
    )
    custom = score_candidate(
        ModelCandidate(
            source="civitai",
            source_id="2",
            name="lora",
            url="u",
            kind="lora",
            description="trained on my photos",
            trigger_words=["ohwx person"],
        )
    )
    assert custom.score > photo_only.score


def test_tier_cuts_list(monkeypatch):
    from kondor_crawler import rank as rank_mod

    monkeypatch.setitem(rank_mod.TIER_LIMITS, "free", 3)
    cands = [
        ModelCandidate(source="hf", source_id=str(i), name=str(i), url="u", score=float(i))
        for i in range(10)
    ]
    cands.sort(key=lambda c: c.score, reverse=True)
    assert len(apply_tier(cands, "free")) == 3


def test_poi_and_stock_agency_boost():
    poi = score_candidate(
        ModelCandidate(
            source="civitai",
            source_id="3",
            name="person",
            url="u",
            kind="lora",
            flags=["poi"],
            trigger_words=["person"],
        )
    )
    stock = score_candidate(
        ModelCandidate(
            source="civitai",
            source_id="4",
            name="pack",
            url="u",
            kind="lora",
            description="trained using Shutterstock photo pack",
            trigger_words=["sks"],
        )
    )
    plain = score_candidate(
        ModelCandidate(
            source="civitai",
            source_id="5",
            name="plain",
            url="u",
            kind="lora",
        )
    )
    assert "poi_likeness" in poi.score_reasons
    assert "custom_data_language" in stock.score_reasons
    assert poi.score > plain.score
    assert stock.score > plain.score


def test_creator_cluster_boosts_siblings():
    from kondor_crawler.rank import rank_candidates

    hot = ModelCandidate(
        source="civitai",
        source_id="1",
        name="a",
        url="u",
        author="alex",
        kind="lora",
        description="DreamBooth trained on my photos",
        trigger_words=["a"],
    )
    sib = ModelCandidate(
        source="civitai",
        source_id="2",
        name="b",
        url="u",
        author="alex",
        kind="lora",
    )
    other = ModelCandidate(
        source="civitai",
        source_id="3",
        name="c",
        url="u",
        author="bob",
        kind="lora",
    )
    ranked = rank_candidates([hot, sib, other])
    by = {c.uid: c for c in ranked}
    assert "creator_cluster" in by["civitai:2"].score_reasons
    assert "creator_cluster" not in by["civitai:3"].score_reasons
    assert by["civitai:2"].score > by["civitai:3"].score


def test_promote_related_by_author():
    a = ModelCandidate(source="civitai", source_id="1", name="a", url="u", author="alex", score=1)
    b = ModelCandidate(source="civitai", source_id="2", name="b", url="u", author="alex", score=1)
    c = ModelCandidate(source="civitai", source_id="3", name="c", url="u", author="bob", score=1)
    out = promote_related([a, b, c], ["civitai:1"], boost=5)
    by_id = {x.uid: x for x in out}
    assert by_id["civitai:2"].score == 6
    assert by_id["civitai:3"].score == 1
