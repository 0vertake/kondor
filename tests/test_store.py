from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.rank import RANKER_VERSION, score_candidate


def _cand(i: str, **kw) -> ModelCandidate:
    defaults = dict(source="huggingface", source_id=i, name=i, url="u", kind="lora")
    defaults.update(kw)
    return ModelCandidate(**defaults)


def test_upsert_new_and_unchanged(store):
    a = _cand("org/a", description="one")
    stats = store.upsert_models([a])
    assert stats["new"] == 1
    stats = store.upsert_models([a])
    assert stats["unchanged"] == 1
    b = _cand("org/a", description="two")
    stats = store.upsert_models([b])
    assert stats["updated"] == 1
    loaded = store.get_model("huggingface:org/a")
    assert loaded is not None
    assert loaded.description == "two"


def test_scores_and_ranked(store):
    c = score_candidate(_cand("org/a", description="DreamBooth trained on my photos"))
    store.upsert_models([c])
    store.save_scores([c], RANKER_VERSION)
    ranked = store.ranked(RANKER_VERSION)
    assert ranked[0].uid == c.uid
    assert ranked[0].score == c.score


def test_crawl_state_and_runs(store):
    run_id = store.start_run(["huggingface"])
    store.set_cursor("huggingface", "12", run_id=run_id)
    st = store.get_crawl_state("huggingface")
    assert st["cursor"] == "12"
    store.mark_source_success("huggingface", "2026-01-01T00:00:00Z", run_id)
    st = store.get_crawl_state("huggingface")
    assert st["cursor"] is None
    assert st["last_success_at"] == "2026-01-01T00:00:00Z"
    store.finish_run(run_id, ok=True, fetched=3, new_count=2)
    last = store.last_successful_run()
    assert last is not None
    assert last["ok"] == 1
    assert last["fetched"] == 3
