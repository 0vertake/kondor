"""Optional per-run enrichment. Never blocks the main crawl past a budget."""

from __future__ import annotations

import re

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.httputil import BudgetExceeded, get_json, get_text
from kondor_crawler.log import get_logger
from kondor_crawler.sources.huggingface import parse_hf_item

log = get_logger("kondor_crawler.enrich")


def enrich_candidate(cand: ModelCandidate) -> ModelCandidate:
    try:
        if cand.source == "huggingface":
            return _enrich_hf(cand)
        if cand.source == "civitai" and not cand.description:
            return _enrich_civitai(cand)
    except BudgetExceeded:
        raise
    except Exception as e:
        log.warning("enrich failed", extra={"data": {"uid": cand.uid, "err": str(e)}})
    return cand


def _enrich_hf(cand: ModelCandidate) -> ModelCandidate:
    item = get_json(f"https://huggingface.co/api/models/{cand.source_id}", params={"full": "true"})
    parsed = parse_hf_item(item)
    if parsed is not None:
        if parsed.description:
            cand.description = parsed.description
        if parsed.claimed_datasets:
            cand.claimed_datasets = list(dict.fromkeys([*cand.claimed_datasets, *parsed.claimed_datasets]))
        if parsed.base_model and not cand.base_model:
            cand.base_model = parsed.base_model
        card = item.get("cardData") or {}
        if isinstance(card, dict):
            for ds in card.get("datasets") or []:
                val = ds if isinstance(ds, str) else ds.get("id")
                if val and val not in cand.claimed_datasets:
                    cand.claimed_datasets.append(val)

    try:
        readme = get_text(f"https://huggingface.co/{cand.source_id}/raw/main/README.md")
    except Exception:
        readme = ""
    if readme:
        if not cand.description:
            cand.description = _plain_readme(readme)[:2000]
        elif len(cand.description) < 80:
            cand.description = (cand.description + "\n" + _plain_readme(readme))[:2000]
        for m in re.finditer(r"datasets:\s*\n((?:\s+-\s+.+\n?)+)", readme, re.I):
            for ds in re.findall(r"-\s+(\S+)", m.group(1)):
                ds = ds.strip("\"'")
                if ds and ds not in cand.claimed_datasets:
                    cand.claimed_datasets.append(ds)
    return cand


def _enrich_civitai(cand: ModelCandidate) -> ModelCandidate:
    data = get_json(f"https://civitai.com/api/v1/models/{cand.source_id}")
    desc = data.get("description") or ""
    if desc:
        cand.description = re.sub(r"<[^>]+>", " ", desc)[:2000]
    versions = data.get("modelVersions") or []
    if versions:
        words = versions[0].get("trainedWords") or []
        if words and not cand.trigger_words:
            cand.trigger_words = list(words)
        notes = versions[0].get("description") or ""
        if notes and notes not in cand.description:
            cand.description = (cand.description + "\n" + re.sub(r"<[^>]+>", " ", notes))[:2000]
    return cand


def _plain_readme(md: str) -> str:
    text = re.sub(r"```.*?```", " ", md, flags=re.S)
    text = re.sub(r"[#*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def should_enrich(cand: ModelCandidate) -> bool:
    if cand.source == "huggingface" and (not cand.description or not cand.claimed_datasets):
        return True
    if cand.source == "civitai" and not cand.description:
        return True
    return False
