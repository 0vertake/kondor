"""Source plugin interface.

Add a new site by implementing `Source` and registering it in `SOURCES`.
"""

from __future__ import annotations

from typing import Callable, Iterator, Protocol

from kondor_crawler.candidate import ModelCandidate

ProgressFn = Callable[[str | None], None]


class Source(Protocol):
    name: str

    def iter_candidates(
        self,
        *,
        limit: int | None = None,
        since: str | None = None,
        start_cursor: str | None = None,
        on_progress: ProgressFn | None = None,
    ) -> Iterator[ModelCandidate]:
        """Yield image fine-tune / LoRA candidates. Newest-first when possible."""


SOURCES: dict[str, type] = {}


def register(source_cls: type) -> type:
    instance_name = getattr(source_cls, "name", None)
    if not instance_name:
        raise ValueError(f"{source_cls!r} needs a class attribute `name`")
    SOURCES[instance_name] = source_cls
    return source_cls


def get_source(name: str, **kwargs) -> Source:
    if name not in SOURCES:
        raise KeyError(f"Unknown source {name!r}. Available: {sorted(SOURCES)}")
    return SOURCES[name](**kwargs)
