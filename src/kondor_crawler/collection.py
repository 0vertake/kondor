"""Client photo-collection domain descriptors for per-client ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Collection:
    """One client's screened photo domain (keywords/tags/description)."""

    id: str
    name: str
    domain_keywords: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    description: str = ""
    tier: str = "standard"
    embedding: list[float] | None = None  # reserved; unused in scoring

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
