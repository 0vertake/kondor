"""Live Hugging Face tests — network required.

These check filter *effectiveness*, not just that the API returns JSON.
"""

from __future__ import annotations

from collections import Counter

import pytest

from kondor_crawler.httputil import get_json
from kondor_crawler.rank import rank_candidates
from kondor_crawler.sources.huggingface import (
    HuggingFaceSource,
    is_probe_eligible,
    parse_hf_item,
)

pytestmark = pytest.mark.live

API = "https://huggingface.co/api/models"
SAMPLE = 80


def _fetch(sort: str, limit: int = SAMPLE) -> list[dict]:
    return get_json(
        API,
        params={
            "pipeline_tag": "text-to-image",
            "sort": sort,
            "direction": -1,
            "limit": limit,
        },
    )


def test_live_schema_and_count():
    cands = list(HuggingFaceSource(page_size=50).iter_candidates(limit=25))
    assert len(cands) == 25
    for c in cands:
        assert c.source == "huggingface"
        assert c.source_id and c.url.startswith("https://huggingface.co/")
        assert c.uid.startswith("huggingface:")
        assert "gguf" not in {t.lower() for t in c.tags} or c.kind == "lora"


def test_live_filter_drops_gguf_quants_on_popular_page():
    page = _fetch("downloads")
    assert len(page) >= 50

    gguf_raw = [
        i
        for i in page
        if "gguf" in {t.lower() for t in i.get("tags") or []}
        or (i.get("library_name") or "").lower() == "gguf"
    ]
    # Popular text-to-image is full of GGUF ports — if HF ever changes, skip softly.
    if not gguf_raw:
        pytest.skip("no GGUF rows in current popular page")

    for item in gguf_raw:
        tags = {t.lower() for t in item.get("tags") or []}
        if tags & {"lora", "peft", "lycoris"}:
            assert is_probe_eligible(item) is True
        else:
            assert is_probe_eligible(item) is False, item.get("id")
            assert parse_hf_item(item) is None


def test_live_source_iterator_emits_no_naked_gguf():
    # Crawl enough that we must have walked past junk on the newest stream.
    cands = list(HuggingFaceSource(page_size=100).iter_candidates(limit=60))
    assert len(cands) == 60
    for c in cands:
        tags = {t.lower() for t in c.tags}
        lib = (c.raw.get("library_name") or "").lower()
        if "gguf" in tags or lib == "gguf":
            assert c.kind == "lora", c.source_id


def test_live_newest_stream_is_mostly_adapters_or_diffusers():
    page = _fetch("lastModified")
    kept = [parse_hf_item(i) for i in page]
    kept = [c for c in kept if c]
    drop_rate = 1 - (len(kept) / len(page))
    kinds = Counter(c.kind for c in kept)

    assert len(kept) >= 40
    # Newest uploads are usually LoRAs / diffusers finetunes, not packaging dumps.
    useful = kinds["lora"] + kinds["finetune"] + kinds["textual_inversion"]
    assert useful / len(kept) >= 0.55, kinds
    # Drop rate should be non-crazy (not filtering everything).
    assert drop_rate < 0.6, drop_rate


def test_live_no_non_image_pipelines_leak_in():
    cands = list(HuggingFaceSource().iter_candidates(limit=40))
    for c in cands:
        pipe = (c.raw.get("pipeline_tag") or "").lower()
        assert pipe in {
            "",
            "text-to-image",
            "image-to-image",
            "unconditional-image-generation",
        }, (c.source_id, pipe)


def test_live_rank_puts_lora_above_untyped_other_when_signals_equalish():
    cands = list(HuggingFaceSource().iter_candidates(limit=40))
    ranked = rank_candidates(cands)
    assert ranked[0].score >= ranked[-1].score
    # Among scored output, mean(lora) should beat mean(other) if both exist.
    by = {}
    for c in ranked:
        by.setdefault(c.kind, []).append(c.score)
    if "lora" in by and "other" in by:
        assert sum(by["lora"]) / len(by["lora"]) > sum(by["other"]) / len(by["other"])


def test_live_parse_roundtrip_ids_stable():
    page = _fetch("lastModified", limit=20)
    for item in page:
        cand = parse_hf_item(item)
        if cand is None:
            continue
        assert cand.source_id == (item.get("modelId") or item["id"])
        assert cand.downloads == int(item.get("downloads") or 0)
