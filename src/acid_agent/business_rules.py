from __future__ import annotations

import re
from collections.abc import Iterable

from acid_agent.models import AcidSubtype

ACID_KEYWORDS = {
    "acid",
    "hcl",
    "hydrochloric",
    "mud acid",
    "hcl/hf",
    "organic acid",
    "acidized",
    "acidizing",
}

ACTION_KEYWORDS = {
    "pump",
    "pumped",
    "perform",
    "performed",
    "job",
    "treatment",
    "stimulation",
    "bullhead",
    "squeeze",
    "spot",
    "circulate",
    "wash",
    "frac",
    "fracture",
}

STRONG_ACTION_KEYWORDS = {
    "pumped",
    "perform",
    "performed",
    "executed",
    "completed",
    "treatment",
    "acid frac",
    "acid wash",
    "matrix acid",
    "stimulation",
}

NEGATIVE_CONTEXT_KEYWORDS = {
    "moved",
    "move",
    "to location",
    "on location",
    "inventory",
    "prepared",
    "tomorrow",
    "planned",
    "plan to",
    "will pump",
}

SUBTYPE_KEYWORDS: dict[AcidSubtype, tuple[str, ...]] = {
    AcidSubtype.ACID_FRACTURING: (
        "acid frac",
        "acid-frac",
        "fracture",
        "frac",
        "etching",
    ),
    AcidSubtype.ACID_WASH: (
        "acid wash",
        "washed",
        "tubing wash",
        "cleanup",
        "scale removal",
        "dissolve scale",
    ),
    AcidSubtype.ACID_SPEARHEAD: (
        "spearhead",
        "preflush",
        "pad acid",
        "spot acid",
        "spotted acid",
    ),
    AcidSubtype.MATRIX_ACIDIZING: (
        "matrix",
        "bullhead",
        "stimulation",
        "near-wellbore",
        "acidize",
        "acidized",
    ),
}

SENTENCE_SPLIT_PATTERN = re.compile(r"[\n\r]+|(?<=[.;!?])\s+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_candidate_sentences(text: str) -> list[str]:
    chunks = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
    return [chunk for chunk in chunks if len(chunk) >= 12]


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def is_acid_job_sentence(text: str) -> bool:
    normalized = normalize_text(text)
    has_acid = _contains_any(normalized, ACID_KEYWORDS)
    has_action = _contains_any(normalized, ACTION_KEYWORDS)
    has_strong_action = _contains_any(normalized, STRONG_ACTION_KEYWORDS)
    has_negative_context = _contains_any(normalized, NEGATIVE_CONTEXT_KEYWORDS)
    if has_negative_context and not has_strong_action:
        return False
    return has_acid and has_action and has_strong_action


def infer_subtype_by_rules(text: str) -> AcidSubtype:
    normalized = normalize_text(text)

    for subtype in (
        AcidSubtype.ACID_FRACTURING,
        AcidSubtype.ACID_WASH,
        AcidSubtype.ACID_SPEARHEAD,
        AcidSubtype.MATRIX_ACIDIZING,
    ):
        if _contains_any(normalized, SUBTYPE_KEYWORDS[subtype]):
            return subtype

    return AcidSubtype.MATRIX_ACIDIZING


def score_sentence_confidence(text: str, subtype: AcidSubtype) -> float:
    normalized = normalize_text(text)

    acid_hit = 1.0 if _contains_any(normalized, ACID_KEYWORDS) else 0.0
    action_hit = 1.0 if _contains_any(normalized, ACTION_KEYWORDS) else 0.0
    subtype_hit = 1.0 if _contains_any(normalized, SUBTYPE_KEYWORDS[subtype]) else 0.0
    detail_hit = 1.0 if re.search(r"\b\d+(?:\.\d+)?\s*(?:bbl|gal|%)\b", normalized) else 0.0

    score = 0.35 * acid_hit + 0.25 * action_hit + 0.30 * subtype_hit + 0.10 * detail_hit
    return min(1.0, max(0.0, score))
