"""GitHub repo discovery for LoRA / DreamBooth releases.

Uses the public Search API (optional GITHUB_TOKEN for higher rate limits).
"""

from __future__ import annotations

import urllib.error
from typing import Any, Iterator

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.httputil import BudgetExceeded, get_json
from kondor_crawler.source import ProgressFn, register

API = "https://api.github.com/search/repositories"
QUERY = "lora OR dreambooth (stable-diffusion OR sdxl OR flux OR photoreal) in:name,description"


def parse_github_item(item: dict[str, Any]) -> ModelCandidate | None:
    full = item.get("full_name") or ""
    if not full:
        return None
    blob = " ".join(
        [
            full,
            item.get("description") or "",
            " ".join(item.get("topics") or []),
        ]
    ).lower()
    if not any(k in blob for k in ("lora", "dreambooth", "stable-diffusion", "sdxl", "flux")):
        return None
    kind = "lora" if "lora" in blob else "finetune"
    author, _, name = full.partition("/")
    return ModelCandidate(
        source="github",
        source_id=full,
        name=name or full,
        url=item.get("html_url") or f"https://github.com/{full}",
        author=author,
        kind=kind,
        tags=list(item.get("topics") or []),
        description=(item.get("description") or "")[:2000],
        likes=int(item.get("stargazers_count") or 0),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at") or item.get("pushed_at"),
        raw={"language": item.get("language"), "fork": bool(item.get("fork"))},
    )


@register
class GitHubSource:
    name = "github"

    def __init__(self, token: str | None = None, page_size: int = 30):
        self.token = token
        self.page_size = min(page_size, 100)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/vnd.github+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def iter_candidates(
        self,
        *,
        limit: int | None = None,
        since: str | None = None,
        start_cursor: str | None = None,
        on_progress: ProgressFn | None = None,
    ) -> Iterator[ModelCandidate]:
        yielded = 0
        page = int(start_cursor) if start_cursor else 1
        while True:
            if limit is not None and yielded >= limit:
                return
            try:
                data = get_json(
                    API,
                    params={
                        "q": QUERY,
                        "sort": "updated",
                        "order": "desc",
                        "per_page": self.page_size,
                        "page": page,
                    },
                    headers=self._headers(),
                )
            except BudgetExceeded:
                return
            except urllib.error.HTTPError as e:
                if e.code in {401, 403, 422}:
                    return
                raise
            items = data.get("items") or []
            if not items:
                return
            older = 0
            for item in items:
                cand = parse_github_item(item)
                if cand is None:
                    continue
                if since and cand.updated_at and cand.updated_at < since:
                    older += 1
                    continue
                yield cand
                yielded += 1
                if limit is not None and yielded >= limit:
                    if on_progress:
                        on_progress(str(page + 1))
                    return
            page += 1
            if on_progress:
                on_progress(str(page))
            if len(items) < self.page_size:
                return
            if since and older == len(items):
                return
            if page > 10:
                # GitHub search caps unauthenticated pagination; stop early.
                return
