"""Pre-probe suspicion ranking.

Inspired by how real systems triage before expensive verification:
- Spawning / Have I Been Trained: prioritize *evidence of specific data*, not popularity
- Civitai taxonomy: Trained vs Merge checkpoints; LoRA + trainedWords = custom-concept prior
- MIA / watermark literature: fine-tunes & adapters are the realistic probe targets;
  mass foundation redistributions are low value for per-pack leakage

Score = P(worth probing for protected-pack leakage), NOT download quality.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from kondor_crawler.candidate import ModelCandidate
from kondor_crawler.collection import Collection

CUSTOM_DATA_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"trained on my",
        r"private (data|dataset|photos|images)",
        r"custom dataset",
        r"my (own )?(photos|images|dataset)",
        r"dream\s*booth",
        r"purchased",
        r"stock (photo|image)",
        r"not public",
        r"dataset from",
        r"personal dataset",
        r"fine-?tuned on \d+",
        r"trained (it )?on \d+",
        r"dataset of \d+",
        r"using \d+ (of )?my",
        r"\b\d{1,3}\s+(images|photos|pics)\b",  # small sets only (1–999)
        # Stock-agency / pack brands (Getty-style shortlist cue)
        r"\bgetty(\s+images)?\b",
        r"\bistock\b",
        r"\bshutterstock\b",
        r"\badobe stock\b",
        r"\balamy\b",
        r"\bunsplash\b",
        r"\bphoto pack\b",
        r"\bimage pack\b",
    ]
]

PHOTO_HINTS = (
    "photo",
    "photoreal",
    "realistic",
    "realism",
    "photograph",
    "product",
    "commercial",
    "portrait",
    "raw photo",
    "studio",
)

MERGE_HINTS = ("merge", "merged", "mix of", "baked", "mergekit")

# Well-known bases / mega community dumps — probe once elsewhere, not every crawl.
FOUNDATION_ID_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"^stabilityai/stable-diffusion",
        r"^stabilityai/sdxl",
        r"^stable-diffusion-v1-5/",
        r"^CompVis/stable-diffusion",
        r"^runwayml/stable-diffusion",
        r"^black-forest-labs/FLUX",
        r"^Comfy-Org/",
        r"^city96/",
        r"^unsloth/",
    ]
]

RANKER_VERSION = "2"

TIER_LIMITS: dict[str, int | None] = {
    "free": 25,
    "standard": 100,
    "pro": 500,
    "enterprise": None,
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+\-/.]{1,}")


def _tokens(text: str) -> set[str]:
    # ponytail: length filter only; stopword lists if noise shows up in eval
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def _blob(cand: ModelCandidate) -> str:
    return " ".join(
        [cand.name, cand.description, " ".join(cand.tags), " ".join(cand.trigger_words)]
    ).lower()


def _is_foundation(cand: ModelCandidate) -> bool:
    if "foundation" in cand.flags:
        return True
    return any(p.search(cand.source_id) for p in FOUNDATION_ID_PATTERNS)


def _has_custom_evidence(cand: ModelCandidate, blob: str) -> bool:
    if cand.claimed_datasets:
        return True
    if any(p.search(blob) for p in CUSTOM_DATA_PATTERNS):
        return True
    # Distinct trigger words ≈ character/product DreamBooth (Civitai-style signal)
    triggers = [t.strip() for t in cand.trigger_words if t and t.strip()]
    if cand.kind in {"lora", "textual_inversion"} and 1 <= len(triggers) <= 12:
        return True
    return False


def score_candidate(cand: ModelCandidate) -> ModelCandidate:
    score = 0.0
    reasons: list[str] = []
    blob = _blob(cand)
    custom = _has_custom_evidence(cand, blob)
    foundation = _is_foundation(cand)
    is_merge = "merge" in cand.flags or any(h in blob for h in MERGE_HINTS)

    # --- adapter / training-type prior (Civitai: LoRA >> Merge) ---
    if cand.kind in {"lora", "textual_inversion"}:
        score += 3.0
        reasons.append(f"kind:{cand.kind}")
    elif cand.kind == "finetune":
        score += 1.0
        reasons.append("kind:finetune")
    elif cand.kind == "checkpoint":
        if "trained_checkpoint" in cand.flags:
            score += 1.5
            reasons.append("kind:trained_checkpoint")
        else:
            score += 0.5
            reasons.append("kind:checkpoint")

    # --- custom-data evidence (Spawning-style: specific data, not vibes) ---
    if any(p.search(blob) for p in CUSTOM_DATA_PATTERNS):
        score += 4.0
        reasons.append("custom_data_language")
        custom = True
    if cand.claimed_datasets:
        score += 3.0
        reasons.append("claimed_dataset")
        custom = True
    triggers = [t.strip() for t in cand.trigger_words if t and t.strip()]
    if cand.kind in {"lora", "textual_inversion"} and 1 <= len(triggers) <= 12:
        score += 2.0
        reasons.append("trigger_words")
        custom = True
    if "poi" in cand.flags:
        score += 2.5
        reasons.append("poi_likeness")
        custom = True

    # Photo domain only matters *with* custom/adapter evidence (fixes Juggernaut problem)
    photo_hits = sum(1 for h in PHOTO_HINTS if h in blob)
    if photo_hits and (custom or cand.kind in {"lora", "textual_inversion"}):
        score += min(1.5, 0.4 * photo_hits)
        reasons.append(f"photo_with_evidence:{photo_hits}")

    # --- demotions ---
    if is_merge:
        score -= 3.0
        reasons.append("merge_penalty")
    if foundation:
        score -= 5.0
        reasons.append("foundation_demote")

    # Popularity is anti-signal for per-pack leakage on non-adapters.
    if cand.kind not in {"lora", "textual_inversion"}:
        if cand.downloads >= 100_000:
            score -= 3.0
            reasons.append("mega_popular_demote")
        elif cand.downloads >= 10_000 and not custom:
            score -= 1.5
            reasons.append("popular_no_custom_demote")

    cand.score = round(score, 3)
    cand.score_reasons = reasons
    return cand


def _apply_creator_cluster(scored: list[ModelCandidate]) -> list[ModelCandidate]:
    evidential_authors: set[str] = set()
    for c in scored:
        if not c.author:
            continue
        if any(
            r in c.score_reasons
            for r in ("custom_data_language", "claimed_dataset", "poi_likeness")
        ):
            evidential_authors.add(c.author.lower())
    for c in scored:
        if c.author and c.author.lower() in evidential_authors:
            if "custom_data_language" not in c.score_reasons and "claimed_dataset" not in c.score_reasons:
                c.score = round(c.score + 1.0, 3)
                c.score_reasons.append("creator_cluster")
    return scored


def rank_candidates(candidates: Iterable[ModelCandidate]) -> list[ModelCandidate]:
    scored = _apply_creator_cluster([score_candidate(c) for c in candidates])
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def domain_relevance(collection: Collection, cand: ModelCandidate) -> tuple[float, list[str]]:
    """Keyword/tag/description overlap between a client collection and a model."""
    phrases = [k.strip().lower() for k in collection.domain_keywords if k and k.strip()]
    kw_tokens: set[str] = set()
    for k in collection.domain_keywords:
        if k:
            kw_tokens |= _tokens(k)
    tag_terms: set[str] = set()
    for t in collection.domain_tags:
        tag_terms |= _tokens(t)
    desc_terms = _tokens(collection.description)

    blob = _blob(cand)
    blob_tokens = _tokens(blob)

    phrase_hits = sum(1 for p in phrases if p and p in blob)
    token_hits = len(kw_tokens & blob_tokens)
    tag_hits = len(tag_terms & blob_tokens)
    desc_hits = len(desc_terms & blob_tokens)

    boost = 0.0
    reasons: list[str] = []
    if phrase_hits:
        boost += min(4.0, 1.5 * phrase_hits)
        reasons.append(f"domain_phrase:{phrase_hits}")
    elif token_hits:
        boost += min(3.0, 0.8 * token_hits)
        reasons.append(f"domain_keyword:{token_hits}")
    if tag_hits:
        boost += min(2.0, 0.7 * tag_hits)
        reasons.append(f"domain_tag:{tag_hits}")
    if desc_hits:
        boost += min(1.5, 0.35 * desc_hits)
        reasons.append(f"domain_desc:{desc_hits}")

    return round(boost, 3), reasons


def rank_for_collection(
    candidates: Iterable[ModelCandidate],
    collection: Collection,
) -> list[ModelCandidate]:
    """Suspicion prior + per-collection domain relevance."""
    scored = _apply_creator_cluster([score_candidate(c) for c in candidates])
    for c in scored:
        boost, reasons = domain_relevance(collection, c)
        if boost > 0:
            c.score = round(c.score + boost, 3)
            c.score_reasons.extend(reasons)
        elif any(
            r in c.score_reasons
            for r in ("custom_data_language", "claimed_dataset", "poi_likeness")
        ):
            c.score = round(c.score - 1.0, 3)
            c.score_reasons.append("domain_miss_strong_custom")
        else:
            c.score = round(c.score - 4.0, 3)
            c.score_reasons.append("domain_miss")
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def apply_tier(
    ranked: list[ModelCandidate],
    tier: str,
    *,
    limits: dict[str, int | None] | None = None,
) -> list[ModelCandidate]:
    limits = limits or TIER_LIMITS
    if tier not in limits:
        raise KeyError(f"Unknown tier {tier!r}. Available: {sorted(limits)}")
    cut = limits[tier]
    return ranked if cut is None else ranked[:cut]


def promote_related(
    ranked: list[ModelCandidate],
    hit_uids: Iterable[str],
    *,
    boost: float = 5.0,
) -> list[ModelCandidate]:
    """Bump same-author siblings after a probe hit."""
    hits = set(hit_uids)
    if not hits:
        return ranked

    by_author: dict[str, list[ModelCandidate]] = defaultdict(list)
    for c in ranked:
        if c.author:
            by_author[c.author.lower()].append(c)

    boost_uids: set[str] = set()
    for c in ranked:
        if c.uid in hits and c.author:
            for sib in by_author[c.author.lower()]:
                if sib.uid not in hits:
                    boost_uids.add(sib.uid)

    for cand in ranked:
        if cand.uid in boost_uids:
            cand.score += boost
            cand.score_reasons.append("promoted_related_hit")
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked
