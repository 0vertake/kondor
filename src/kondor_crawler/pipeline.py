"""Crawl → dedupe → rank → tier cut → JSONL.

Probe / watermark code should consume the JSONL (or call `run_crawl` in-process).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

# Ensure plugins register.
import kondor_crawler.sources  # noqa: F401
from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.rank import apply_tier, rank_candidates
from kondor_crawler.source import SOURCES, get_source


def crawl(
    sources: Sequence[str] | None = None,
    *,
    limit_per_source: int | None = None,
    hf_token: str | None = None,
    civitai_token: str | None = None,
    github_token: str | None = None,
) -> list[ModelCandidate]:
    names = list(sources) if sources is not None else sorted(SOURCES)
    found: list[ModelCandidate] = []
    for name in names:
        kwargs: dict = {}
        if name == "huggingface" and hf_token:
            kwargs["token"] = hf_token
        if name == "civitai" and civitai_token:
            kwargs["token"] = civitai_token
        if name == "github" and github_token:
            kwargs["token"] = github_token
        src = get_source(name, **kwargs)
        found.extend(src.iter_candidates(limit=limit_per_source))
    return dedupe(found)


def dedupe(candidates: Iterable[ModelCandidate]) -> list[ModelCandidate]:
    seen: set[str] = set()
    out: list[ModelCandidate] = []
    for c in candidates:
        if c.uid in seen:
            continue
        seen.add(c.uid)
        out.append(c)
    return out


def write_jsonl(path: Path | str, candidates: Iterable[ModelCandidate]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[ModelCandidate]:
    path = Path(path)
    out: list[ModelCandidate] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ModelCandidate.from_dict(json.loads(line)))
    return out


def run_crawl(
    *,
    sources: Sequence[str] | None = None,
    limit_per_source: int | None = None,
    tier: str | None = "standard",
    out_path: Path | str | None = None,
    hf_token: str | None = None,
) -> list[ModelCandidate]:
    raw = crawl(sources, limit_per_source=limit_per_source, hf_token=hf_token)
    ranked = rank_candidates(raw)
    selected = apply_tier(ranked, tier) if tier else ranked
    if out_path is not None:
        write_jsonl(out_path, selected)
    return selected
