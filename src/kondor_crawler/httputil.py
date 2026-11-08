"""JSON GET with retries, Retry-After, min-interval, and a per-run budget."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_HEADERS = {
    "User-Agent": "kondor-crawler/0.1 (+https://github.com/kondor)",
    "Accept": "application/json",
}


class BudgetExceeded(RuntimeError):
    pass


class HttpClient:
    def __init__(self, *, min_interval: float = 0.2, budget: int | None = None):
        self.min_interval = min_interval
        self.budget = budget
        self.used = 0
        self._last = 0.0

    def remaining(self) -> int | None:
        if self.budget is None:
            return None
        return max(0, self.budget - self.used)

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        timeout: float = 30.0,
    ) -> Any:
        body = self._get_bytes(url, params=params, headers=headers, retries=retries, timeout=timeout)
        return json.loads(body.decode("utf-8"))

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        timeout: float = 30.0,
    ) -> str:
        body = self._get_bytes(url, params=params, headers=headers, retries=retries, timeout=timeout)
        return body.decode("utf-8", errors="replace")

    def _get_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        retries: int,
        timeout: float,
    ) -> bytes:
        if self.budget is not None and self.used >= self.budget:
            raise BudgetExceeded(f"HTTP budget {self.budget} exhausted")
        if params:
            pairs: list[tuple[str, str]] = []
            for key, value in params.items():
                if isinstance(value, (list, tuple)):
                    for item in value:
                        pairs.append((key, str(item)))
                elif value is not None:
                    pairs.append((key, str(value)))
            url = f"{url}?{urllib.parse.urlencode(pairs)}"

        req = urllib.request.Request(url, headers={**DEFAULT_HEADERS, **(headers or {})})
        last_err: Exception | None = None
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    self._last = time.monotonic()
                    self.used += 1
                    return resp.read()
            except urllib.error.HTTPError as e:
                last_err = e
                self._last = time.monotonic()
                self.used += 1
                if e.code in {429, 500, 502, 503, 504} and attempt < retries - 1:
                    time.sleep(_retry_after(e, attempt))
                    continue
                raise
            except urllib.error.URLError as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError(f"GET failed for {url}: {last_err}")


def _retry_after(err: urllib.error.HTTPError, attempt: int) -> float:
    raw = err.headers.get("Retry-After") if err.headers else None
    if raw:
        try:
            return min(float(raw), 60.0)
        except ValueError:
            pass
    return 2**attempt


# Module-level client so existing call sites keep working.
http = HttpClient()


def configure_http(*, min_interval: float = 0.2, budget: int | None = None) -> HttpClient:
    global http
    http = HttpClient(min_interval=min_interval, budget=budget)
    return http


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    timeout: float = 30.0,
) -> Any:
    return http.get_json(url, params=params, headers=headers, retries=retries, timeout=timeout)


def get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    timeout: float = 30.0,
) -> str:
    return http.get_text(url, params=params, headers=headers, retries=retries, timeout=timeout)
