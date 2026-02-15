from __future__ import annotations

import re
from collections.abc import Iterable

from ops_assistant.models import EventSubtype

TRIGGER_KEYWORDS = {
    "chemical",
    "treatment",
    "intervention",
    "service fluid",
    "blend",
    "additive",
}

ACTION_KEYWORDS = {
    "pump",
    "pumped",
    "perform",
    "performed",
    "job",
    "operation",
    "execute",
    "executed",
    "stage",
    "circulate",
    "flush",
    "inject",
}

STRONG_ACTION_KEYWORDS = {
    "pumped",
    "perform",
    "performed",
    "executed",
    "completed",
    "operation",
    "stage",
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

SUBTYPE_KEYWORDS: dict[EventSubtype, tuple[str, ...]] = {
    EventSubtype.CATEGORY_ALPHA: (
        "alpha",
    ),
    EventSubtype.CATEGORY_BETA: (
        "beta",
    ),
    EventSubtype.CATEGORY_GAMMA: (
        "gamma",
    ),
    EventSubtype.CATEGORY_DELTA: (
        "delta",
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


def is_target_event_sentence(text: str) -> bool:
    normalized = normalize_text(text)
    has_trigger = _contains_any(normalized, TRIGGER_KEYWORDS)
    has_action = _contains_any(normalized, ACTION_KEYWORDS)
    has_strong_action = _contains_any(normalized, STRONG_ACTION_KEYWORDS)
    has_negative_context = _contains_any(normalized, NEGATIVE_CONTEXT_KEYWORDS)
    if has_negative_context and not has_strong_action:
        return False
    return has_trigger and has_action and has_strong_action


def infer_category_by_rules(text: str) -> EventSubtype:
    normalized = normalize_text(text)

    for subtype in (
        EventSubtype.CATEGORY_ALPHA,
        EventSubtype.CATEGORY_BETA,
        EventSubtype.CATEGORY_GAMMA,
        EventSubtype.CATEGORY_DELTA,
    ):
        if _contains_any(normalized, SUBTYPE_KEYWORDS[subtype]):
            return subtype

    return EventSubtype.CATEGORY_ALPHA


def score_sentence_confidence(text: str, subtype: EventSubtype) -> float:
    normalized = normalize_text(text)

    trigger_hit = 1.0 if _contains_any(normalized, TRIGGER_KEYWORDS) else 0.0
    action_hit = 1.0 if _contains_any(normalized, ACTION_KEYWORDS) else 0.0
    subtype_hit = 1.0 if _contains_any(normalized, SUBTYPE_KEYWORDS[subtype]) else 0.0
    detail_hit = 1.0 if re.search(r"\b\d+(?:\.\d+)?\s*(?:bbl|gal|%|psi)\b", normalized) else 0.0

    score = 0.35 * trigger_hit + 0.25 * action_hit + 0.30 * subtype_hit + 0.10 * detail_hit
    return min(1.0, max(0.0, score))
