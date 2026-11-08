import json
from pathlib import Path

from kondor_crawler.sources.github import parse_github_item

FIX = Path(__file__).parent / "fixtures" / "github_search.json"


def test_parses_lora_repo_skips_unrelated():
    data = json.loads(FIX.read_text())
    parsed = [parse_github_item(i) for i in data["items"]]
    kept = [c for c in parsed if c]
    assert len(kept) == 1
    assert kept[0].source_id == "alice/product-lora"
    assert kept[0].kind == "lora"
    assert kept[0].author == "alice"
