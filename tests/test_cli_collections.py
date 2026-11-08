import json

import pytest

from kondor_crawler.cli import main


def test_unknown_collection_exits(monkeypatch, store):
    monkeypatch.setenv("DATABASE_URL", store.url)
    with pytest.raises(SystemExit) as ei:
        main(["enqueue", "--collection", "does-not-exist"])
    assert "unknown collection" in str(ei.value)


def test_collection_upsert_and_list(monkeypatch, store, capsys):
    monkeypatch.setenv("DATABASE_URL", store.url)
    assert main([
        "collection",
        "upsert",
        "--id",
        "acme",
        "--keywords",
        "product photography",
        "--tags",
        "commercial",
        "--tier",
        "standard",
    ]) == 0
    capsys.readouterr()  # discard upsert JSON
    assert main(["collection", "list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["id"] == "acme"
    assert "product photography" in rows[0]["keywords"]


def test_empty_domain_warns(monkeypatch, store, capsys):
    monkeypatch.setenv("DATABASE_URL", store.url)
    assert main(["collection", "upsert", "--id", "empty", "--tier", "free"]) == 0
    err = capsys.readouterr().err
    assert "warning" in err.lower()
    assert "domain" in err.lower()
