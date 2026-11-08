"""Stage 4 core: suspect intake. The crawler's real format is unknown (decision D5), so this is the
adapter's target record. Everything we cannot test is logged as 'not testable', never as clean."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

SUPPORTED_FAMILIES = {"pixel-diffusion"}      # latent diffusion needs its own engine (decision D3)


@dataclass
class Suspect:
    suspect_id: str
    source_url: str
    access: str                  # "open-weights" or "api"
    family: str                  # "pixel-diffusion", "latent-diffusion", "other"
    claimed_base_model: str | None = None
    claimed_training_data: str | None = None
    license: str | None = None
    discovered_at: str | None = None

    def as_dict(self):
        return asdict(self)


def eligibility(s: Suspect, image_size: int | None = None) -> tuple[bool, str]:
    if s.access != "open-weights":
        return False, f"access is '{s.access}': querying an API needs a terms decision (D4)"
    if s.family not in SUPPORTED_FAMILIES:
        return False, f"family '{s.family}' has no trained configuration"
    if not s.claimed_base_model:
        return False, "no claimed base model, so there is no clean reference for test 2"
    return True, "eligible"


def read_suspects(path: Path) -> list[Suspect]:
    return [Suspect(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]
