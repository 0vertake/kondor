"""Hugging Face model discovery (text-to-image)."""

from __future__ import annotations

from typing import Any, Iterator

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.httputil import BudgetExceeded, get_json
from kondor_crawler.source import ProgressFn, register

API = "https://huggingface.co/api/models"

# Packaging / runtime ports — not new training runs (unless also tagged as LoRA).
_JUNK_FORMATS = frozenset(
    {"gguf", "ggml", "onnx", "openvino", "tflite", "coreml", "tensorrt", "ncnn"}
)


def _tags_lower(tags: list[str]) -> set[str]:
    return {t.lower() for t in tags}


def kind_from_item(item: dict[str, Any]) -> str:
    tags = _tags_lower(item.get("tags") or [])
    if "lora" in tags or "peft" in tags or "lycoris" in tags:
        return "lora"
    if "textual-inversion" in tags or "embedding" in tags:
        return "textual_inversion"
    if "diffusers" in tags or (item.get("library_name") or "").lower() == "diffusers":
        return "finetune"
    if "diffusion-single-file" in tags or "comfyui" in tags:
        return "checkpoint"
    return "other"


def base_from_tags(tags: list[str]) -> str:
    for tag in tags:
        if tag.startswith("base_model:") and "adapter:" not in tag and "quantized:" not in tag:
            return tag.split(":", 1)[1]
    return ""


def dataset_tags(tags: list[str]) -> list[str]:
    return [t.split(":", 1)[1] for t in tags if t.lower().startswith("dataset:")]


def is_probe_eligible(item: dict[str, Any]) -> bool:
    """Keep models that could be fine-tunes carrying training data.

    Drops pure quant / runtime re-packs (GGUF packs, ONNX-primary ports, …).
    An `onnx` *tag* on a diffusers model is fine — HF often lists export formats.
    # ponytail: tag/library heuristics; tighten with known-base blocklist when probe budget hurts.
    """
    tags = _tags_lower(item.get("tags") or [])
    library = (item.get("library_name") or "").lower()
    pipeline = (item.get("pipeline_tag") or "").lower()

    if pipeline and pipeline not in {
        "text-to-image",
        "image-to-image",
        "unconditional-image-generation",
    }:
        return False

    # Adapters always eligible (even if a GGUF sibling file is also tagged).
    if tags & {"lora", "peft", "lycoris", "textual-inversion", "embedding"}:
        return True

    if library in _JUNK_FORMATS:
        return False
    if tags & {"gguf", "ggml"}:
        return False
    if "quantized" in tags and "diffusers" not in tags:
        return False

    return True


_FOUNDATION_PREFIXES = (
    "stabilityai/stable-diffusion",
    "stabilityai/sdxl",
    "stable-diffusion-v1-5/",
    "CompVis/stable-diffusion",
    "runwayml/stable-diffusion",
    "black-forest-labs/FLUX",
    "Comfy-Org/",
    "city96/",
    "unsloth/",
)


def parse_hf_item(item: dict[str, Any]) -> ModelCandidate | None:
    if not is_probe_eligible(item):
        return None
    tags = list(item.get("tags") or [])
    model_id = str(item.get("modelId") or item.get("id") or "")
    if not model_id:
        return None
    author, _, name = model_id.partition("/")
    if not name:
        author, name = "", model_id

    datasets = dataset_tags(tags)
    card = item.get("cardData") or {}
    if isinstance(card, dict):
        for ds in card.get("datasets") or []:
            if isinstance(ds, str) and ds not in datasets:
                datasets.append(ds)
            elif isinstance(ds, dict) and ds.get("id") and ds["id"] not in datasets:
                datasets.append(ds["id"])

    flags: list[str] = []
    if any(model_id.startswith(p) or model_id.lower().startswith(p.lower()) for p in _FOUNDATION_PREFIXES):
        flags.append("foundation")

    return ModelCandidate(
        source="huggingface",
        source_id=model_id,
        name=name,
        url=f"https://huggingface.co/{model_id}",
        author=author,
        kind=kind_from_item(item),
        base_model=base_from_tags(tags) or str(card.get("base_model") or ""),
        tags=tags,
        downloads=int(item.get("downloads") or 0),
        likes=int(item.get("likes") or 0),
        created_at=item.get("createdAt"),
        updated_at=item.get("lastModified"),
        claimed_datasets=datasets,
        flags=flags,
        raw={
            "pipeline_tag": item.get("pipeline_tag"),
            "library_name": item.get("library_name"),
        },
    )


@register
class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, token: str | None = None, page_size: int = 100):
        self.token = token
        self.page_size = page_size

    def iter_candidates(
        self,
        *,
        limit: int | None = None,
        since: str | None = None,
        start_cursor: str | None = None,
        on_progress: ProgressFn | None = None,
    ) -> Iterator[ModelCandidate]:
        yielded = 0
        offset = int(start_cursor) if start_cursor else 0
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        while True:
            if limit is not None and yielded >= limit:
                return
            try:
                data = get_json(
                    API,
                    params={
                        "pipeline_tag": "text-to-image",
                        "sort": "lastModified",
                        "direction": -1,
                        "limit": self.page_size,
                        "offset": offset,
                    },
                    headers=headers,
                )
            except BudgetExceeded:
                return
            if not isinstance(data, list) or not data:
                return

            older_than_since = 0
            for item in data:
                cand = parse_hf_item(item)
                if cand is None:
                    ts = item.get("lastModified") or item.get("createdAt")
                    if since and ts and ts < since:
                        older_than_since += 1
                    continue
                if since and cand.updated_at and cand.updated_at < since:
                    older_than_since += 1
                    continue
                yield cand
                yielded += 1
                if limit is not None and yielded >= limit:
                    if on_progress:
                        on_progress(str(offset + len(data)))
                    return

            offset += len(data)
            if on_progress:
                on_progress(str(offset))
            if len(data) < self.page_size:
                return
            if since and older_than_since == len(data):
                return
