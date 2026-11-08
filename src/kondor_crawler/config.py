"""Env-only settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


@dataclass(slots=True)
class Settings:
    database_url: str = ""
    hf_token: str | None = None
    civitai_token: str | None = None
    github_token: str | None = None
    request_budget: int = 400
    enrich_budget: int = 40
    min_interval: float = 0.2
    page_size: int = 100
    default_tier: str = "standard"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.environ.get("DATABASE_URL", ""),
            hf_token=os.environ.get("HF_TOKEN") or None,
            civitai_token=os.environ.get("CIVITAI_TOKEN") or None,
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            request_budget=_int("CRAWL_REQUEST_BUDGET", 400),
            enrich_budget=_int("ENRICH_BUDGET", 40),
            min_interval=_float("CRAWL_MIN_INTERVAL", 0.2),
            page_size=_int("CRAWL_PAGE_SIZE", 100),
            default_tier=os.environ.get("CRAWL_TIER", "standard"),
        )
