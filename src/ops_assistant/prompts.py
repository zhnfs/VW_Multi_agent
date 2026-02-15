from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

RULES_FILE = Path(__file__).parent / "resources" / "sme_rules_placeholder.txt"


def load_sme_business_rules() -> str:
    return RULES_FILE.read_text(encoding="utf-8")


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a planning agent for event-report QA. "
            "Return JSON only with keys intent and asset_id. "
            "intent must be one of: count, subtypes, both.",
        ),
        (
            "human",
            "Question: {question}\n"
            "Optional explicit asset id: {asset_id}\n"
            "Infer intent and asset_id from the question when needed.",
        ),
    ]
)


CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify event text into exactly one subtype: "
            "category_alpha, category_beta, category_gamma, category_delta. "
            "Return JSON only with keys subtype and reasoning_short.",
        ),
        (
            "human",
            "Business rules:\n{rules}\n\nReport evidence:\n{evidence}",
        ),
    ]
)


VALIDATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict factuality checker. "
            "Compare candidate answer against evidence and return JSON only "
            "with supported_claims, unsupported_claims, and faithful_score (0 to 1).",
        ),
        (
            "human",
            "Question: {question}\nCandidate answer:\n{answer}\nEvidence:\n{evidence}",
        ),
    ]
)
