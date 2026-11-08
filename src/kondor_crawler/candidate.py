"""Shared candidate schema — the contract between crawl and probe."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ModelCandidate:
    """One discoverable image model / LoRA / fine-tune.

    Probe teammates should treat this as the input record. Ranking fields are
    filled by the crawler; watermark results belong elsewhere.
    """

    source: str  # e.g. "huggingface", "civitai"
    source_id: str  # stable id within that source
    name: str
    url: str
    author: str = ""
    kind: str = ""  # lora | checkpoint | textual_inversion | finetune | other
    base_model: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""
    trigger_words: list[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    preview_urls: list[str] = field(default_factory=list)
    claimed_datasets: list[str] = field(default_factory=list)
    # Free-form flags from sources: merge | trained_checkpoint | foundation | …
    flags: list[str] = field(default_factory=list)
    score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)
    discovered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.source_id}"

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "kind": self.kind,
                "description": self.description,
                "tags": self.tags,
                "trigger_words": self.trigger_words,
                "claimed_datasets": self.claimed_datasets,
                "flags": self.flags,
                "downloads": self.downloads,
                "likes": self.likes,
                "author": self.author,
                "base_model": self.base_model,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["uid"] = self.uid
        data["content_hash"] = self.content_hash()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCandidate:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
