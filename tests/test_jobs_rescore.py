from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.jobs import rescore
from kondor_crawler.rank import RANKER_VERSION


def test_rescore_promotes_hit_author_siblings(store):
    a = ModelCandidate(source="civitai", source_id="1", name="a", url="u", author="alex", kind="lora")
    b = ModelCandidate(source="civitai", source_id="2", name="b", url="u", author="alex", kind="lora")
    c = ModelCandidate(source="civitai", source_id="3", name="c", url="u", author="bob", kind="lora")
    store.upsert_models([a, b, c])
    store.record_result("civitai:1", "hit")
    rescore(store)
    ranked = store.ranked(RANKER_VERSION)
    by = {x.uid: x for x in ranked}
    assert "promoted_related_hit" in by["civitai:2"].score_reasons
    assert "promoted_related_hit" not in by["civitai:3"].score_reasons
    assert by["civitai:2"].score > by["civitai:3"].score
