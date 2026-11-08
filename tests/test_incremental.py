from kondor_crawler.httputil import BudgetExceeded, HttpClient, configure_http
from kondor_crawler.sources.huggingface import HuggingFaceSource


def test_hf_stops_at_since(monkeypatch):
    pages = [
        [
            {
                "id": "org/new",
                "modelId": "org/new",
                "tags": ["lora", "text-to-image"],
                "pipeline_tag": "text-to-image",
                "lastModified": "2026-06-01T00:00:00.000Z",
                "createdAt": "2026-06-01T00:00:00.000Z",
                "downloads": 0,
                "likes": 0,
            },
            {
                "id": "org/old",
                "modelId": "org/old",
                "tags": ["lora", "text-to-image"],
                "pipeline_tag": "text-to-image",
                "lastModified": "2024-01-01T00:00:00.000Z",
                "createdAt": "2024-01-01T00:00:00.000Z",
                "downloads": 0,
                "likes": 0,
            },
        ]
    ]

    def fake_json(*_a, **_k):
        return pages.pop(0) if pages else []

    monkeypatch.setattr("kondor_crawler.sources.huggingface.get_json", fake_json)
    src = HuggingFaceSource(page_size=10)
    got = list(src.iter_candidates(since="2025-01-01T00:00:00.000Z"))
    assert [c.source_id for c in got] == ["org/new"]


def test_hf_persists_cursor(monkeypatch):
    def fake_json(*_a, **kw):
        offset = (kw.get("params") or {}).get("offset", 0)
        item = {
            "id": f"org/m{offset}",
            "modelId": f"org/m{offset}",
            "tags": ["lora", "text-to-image"],
            "pipeline_tag": "text-to-image",
            "lastModified": "2026-06-01T00:00:00.000Z",
            "downloads": 0,
            "likes": 0,
        }
        return [item]

    monkeypatch.setattr("kondor_crawler.sources.huggingface.get_json", fake_json)
    src = HuggingFaceSource(page_size=1)
    cursors = []
    list(src.iter_candidates(limit=2, on_progress=cursors.append))
    assert cursors
    assert cursors[-1] == "2"


def test_budget_stops_requests():
    client = configure_http(min_interval=0, budget=0)
    try:
        client.get_json("http://example.invalid")
        raise AssertionError("expected BudgetExceeded")
    except BudgetExceeded:
        pass
    configure_http(min_interval=0.2, budget=None)  # reset module client for other tests


def test_http_client_counts_budget():
    c = HttpClient(min_interval=0, budget=1)
    assert c.remaining() == 1
    # don't actually fetch; remaining math
    c.used = 1
    assert c.remaining() == 0
