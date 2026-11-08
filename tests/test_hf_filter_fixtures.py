"""Offline filter tests against real Hugging Face API payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kondor_crawler.rank import rank_candidates
from kondor_crawler.sources.huggingface import is_probe_eligible, parse_hf_item

FIX = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIX / name).read_text())


def test_rejects_known_gguf_quants():
    for item in _load("hf_named_junk.json"):
        mid = item.get("id") or item.get("modelId")
        if "gguf" in mid.lower() or "gguf" in {t.lower() for t in item.get("tags") or []}:
            assert is_probe_eligible(item) is False, mid
            assert parse_hf_item(item) is None, mid


def test_accepts_lora_search_hits():
    items = _load("hf_lora_search.json")
    assert items
    accepted = [parse_hf_item(i) for i in items]
    assert all(c is not None for c in accepted)
    assert all(c.kind == "lora" for c in accepted)
    assert all(c.source == "huggingface" for c in accepted)


def test_downloads_page_drops_gguf_keeps_adapters_and_diffusers():
    page = _load("hf_downloads_page.json")
    parsed = [parse_hf_item(i) for i in page]
    kept = [c for c in parsed if c is not None]
    dropped_ids = [i["id"] for i, c in zip(page, parsed) if c is None]

    assert dropped_ids, "expected some quant/junk drops on popular page"
    for did in dropped_ids:
        item = next(i for i in page if i["id"] == did)
        assert is_probe_eligible(item) is False
        tags = {t.lower() for t in item.get("tags") or []}
        lib = (item.get("library_name") or "").lower()
        assert (
            lib in {"gguf", "ggml", "onnx", "openvino", "tflite", "coreml", "tensorrt", "ncnn"}
            or tags & {"gguf", "ggml"}
            or ("quantized" in tags and "diffusers" not in tags)
        ), did

    for c in kept:
        tags = {t.lower() for t in c.tags}
        if "gguf" in tags:
            assert c.kind == "lora"


def test_newest_page_majority_probe_eligible_and_typed():
    page = _load("hf_newest_page.json")
    kept = [parse_hf_item(i) for i in page]
    kept = [c for c in kept if c]
    assert len(kept) >= int(0.7 * len(page))
    typed = [c for c in kept if c.kind in {"lora", "finetune", "textual_inversion", "checkpoint"}]
    assert len(typed) / len(kept) >= 0.7


def test_rank_separates_custom_photo_lora_from_merge_and_quant_noise():
    from kondor_crawler.candidate import ModelCandidate

    custom = ModelCandidate(
        source="huggingface",
        source_id="studio/product-lora",
        name="product-lora",
        url="u",
        kind="lora",
        description="DreamBooth trained on my photos, private dataset of product shots",
        tags=["lora", "photorealistic", "product", "text-to-image"],
    )
    merge = ModelCandidate(
        source="huggingface",
        source_id="x/anime-merge",
        name="anime-merge",
        url="u",
        kind="checkpoint",
        description="Merged mix of baked anime checkpoints",
        tags=["merge", "anime", "text-to-image"],
    )
    ranked = rank_candidates([merge, custom])
    assert ranked[0].source_id == "studio/product-lora"
    assert ranked[0].score > ranked[1].score


@pytest.mark.parametrize(
    "item,eligible",
    [
        (
            {
                "id": "org/my-lora",
                "pipeline_tag": "text-to-image",
                "tags": ["lora", "text-to-image", "gguf"],  # adapter wins
            },
            True,
        ),
        (
            {
                "id": "org/base-gguf",
                "pipeline_tag": "text-to-image",
                "library_name": "gguf",
                "tags": ["gguf", "quantized", "text-to-image"],
            },
            False,
        ),
        (
            {
                "id": "org/nlp",
                "pipeline_tag": "text-generation",
                "tags": ["text-generation"],
            },
            False,
        ),
        (
            {
                "id": "org/sdxl-ft",
                "pipeline_tag": "text-to-image",
                "library_name": "diffusers",
                "tags": ["diffusers", "text-to-image", "safetensors"],
            },
            True,
        ),
    ],
)
def test_eligibility_matrix(item, eligible):
    assert is_probe_eligible(item) is eligible
    cand = parse_hf_item(item)
    assert (cand is not None) is eligible
