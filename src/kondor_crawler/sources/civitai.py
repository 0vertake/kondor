"""Civitai model discovery.

Uses Civitai's own Trained vs Merge checkpoint split (see developer.civitai.com
models API `checkpointType`) so merges aren't mixed with real fine-tunes.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.httputil import BudgetExceeded, get_json
from kondor_crawler.source import ProgressFn, register

API = "https://civitai.com/api/v1/models"

_KIND = {
    "LORA": "lora",
    "LoCon": "lora",
    "Checkpoint": "checkpoint",
    "TextualInversion": "textual_inversion",
}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _join_cursor(label: str, cursor: str | None) -> str | None:
    if not cursor:
        return None
    return f"{label}:{cursor}"


def _split_cursor(start: str | None) -> tuple[str | None, str | None]:
    if not start or ":" not in start:
        return None, start
    label, _, rest = start.partition(":")
    return label, rest or None


@register
class CivitaiSource:
    name = "civitai"

    def __init__(
        self,
        page_size: int = 100,
        include_merges: bool = False,
        token: str | None = None,
    ):
        self.page_size = page_size
        self.include_merges = include_merges
        self.token = token

    def iter_candidates(
        self,
        *,
        limit: int | None = None,
        since: str | None = None,
        start_cursor: str | None = None,
        on_progress: ProgressFn | None = None,
    ) -> Iterator[ModelCandidate]:
        yielded = 0
        streams: list[dict[str, Any]] = [
            {"types": ["LORA", "LoCon", "TextualInversion"], "label": "adapter"},
            {"types": ["Checkpoint"], "checkpointType": "Trained", "label": "trained"},
        ]
        if self.include_merges:
            streams.append(
                {"types": ["Checkpoint"], "checkpointType": "Merge", "label": "merge"}
            )

        # start_cursor encodes "label:cursor" so we resume the right stream.
        resume_label, resume_cur = _split_cursor(start_cursor)
        started = resume_label is None
        for stream in streams:
            if not started:
                if stream["label"] != resume_label:
                    continue
                started = True
            if limit is not None and yielded >= limit:
                return
            remaining = None if limit is None else limit - yielded
            for cand in self._iter_stream(
                stream,
                limit=remaining,
                since=since,
                start_cursor=resume_cur if stream["label"] == resume_label else None,
                on_progress=on_progress,
            ):
                yield cand
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            resume_cur = None

    def _headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _iter_stream(
        self,
        stream: dict[str, Any],
        *,
        limit: int | None,
        since: str | None = None,
        start_cursor: str | None = None,
        on_progress: ProgressFn | None = None,
    ) -> Iterator[ModelCandidate]:
        yielded = 0
        cursor: str | None = start_cursor
        label = stream["label"]
        while True:
            if limit is not None and yielded >= limit:
                return
            params: dict[str, Any] = {
                "types": stream["types"],
                "sort": "Newest",
                "limit": self.page_size if limit is None else min(self.page_size, limit - yielded),
            }
            if "checkpointType" in stream:
                params["checkpointType"] = stream["checkpointType"]
            if cursor:
                params["cursor"] = cursor

            try:
                data = get_json(API, params=params, headers=self._headers())
            except BudgetExceeded:
                return
            items = data.get("items") or []
            if not items:
                return

            older = 0
            for item in items:
                cand = self._to_candidate(item, stream_label=label)
                if since and cand.updated_at and cand.updated_at < since:
                    older += 1
                    continue
                yield cand
                yielded += 1
                if limit is not None and yielded >= limit:
                    nxt = (data.get("metadata") or {}).get("nextCursor")
                    if on_progress:
                        on_progress(_join_cursor(label, nxt))
                    return

            cursor = (data.get("metadata") or {}).get("nextCursor")
            if on_progress:
                on_progress(_join_cursor(label, cursor))
            if not cursor:
                return
            if since and older == len(items):
                return

    def _to_candidate(self, item: dict[str, Any], *, stream_label: str) -> ModelCandidate:
        versions = item.get("modelVersions") or []
        latest = versions[0] if versions else {}
        base = latest.get("baseModel") or (item.get("baseModels") or [""])[0]
        previews = [img["url"] for img in (latest.get("images") or []) if img.get("url")]
        stats = item.get("stats") or {}
        flags: list[str] = []
        if stream_label == "merge":
            flags.append("merge")
        elif stream_label == "trained":
            flags.append("trained_checkpoint")
        # Civitai likeness/real-person flag — strong private-photo prior
        if item.get("poi"):
            flags.append("poi")

        return ModelCandidate(
            source=self.name,
            source_id=str(item.get("id")),
            name=item.get("name") or str(item.get("id")),
            url=f"https://civitai.com/models/{item.get('id')}",
            author=(item.get("creator") or {}).get("username") or "",
            kind=_KIND.get(item.get("type") or "", "other"),
            base_model=str(base or ""),
            tags=list(item.get("tags") or []),
            description=_strip_html(item.get("description") or "")[:2000],
            trigger_words=list(latest.get("trainedWords") or []),
            downloads=int(stats.get("downloadCount") or 0),
            likes=int(stats.get("thumbsUpCount") or 0),
            created_at=latest.get("publishedAt"),
            updated_at=latest.get("publishedAt"),
            preview_urls=previews[:5],
            flags=flags,
            raw={
                "type": item.get("type"),
                "version_id": latest.get("id"),
                "stream": stream_label,
                "poi": bool(item.get("poi")),
            },
        )
