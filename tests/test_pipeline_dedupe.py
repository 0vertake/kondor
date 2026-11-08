from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.pipeline import dedupe, read_jsonl, write_jsonl


def test_dedupe_by_uid(tmp_path):
    a = ModelCandidate(source="huggingface", source_id="org/m", name="m", url="u")
    b = ModelCandidate(source="huggingface", source_id="org/m", name="m", url="u")
    c = ModelCandidate(source="civitai", source_id="9", name="x", url="u")
    assert [x.uid for x in dedupe([a, b, c])] == ["huggingface:org/m", "civitai:9"]

    path = tmp_path / "c.jsonl"
    write_jsonl(path, [a, c])
    loaded = read_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].source_id == "org/m"
