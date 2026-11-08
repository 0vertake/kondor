from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.eval import average_precision, hit_rate_at_k, precision_at_k, summarize


def _c(i, score):
    x = ModelCandidate(source="hf", source_id=str(i), name=str(i), url="u")
    x.score = score
    return x


def test_precision_and_ap():
    ranked = [_c(1, 3), _c(2, 2), _c(3, 1)]
    labels = {"hf:1": "worthy", "hf:2": "skip", "hf:3": "worthy"}
    p, n = precision_at_k(ranked, labels, 2)
    assert n == 2
    assert p == 0.5
    ap = average_precision(ranked, labels)
    assert ap is not None
    assert ap > 0


def test_hit_rate_and_summarize():
    ranked = [_c(1, 3), _c(2, 2), _c(3, 1)]
    hits = {"hf:1"}
    assert hit_rate_at_k((c.uid for c in ranked), hits, 2) == 0.5
    report = summarize(ranked, hits=hits)
    assert report["hit_rate_at"][10] == 1 / 3
