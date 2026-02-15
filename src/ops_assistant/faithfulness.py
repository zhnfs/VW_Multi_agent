from __future__ import annotations

import json
from statistics import mean

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ops_assistant.models import EventRecord
from ops_assistant.prompts import VALIDATOR_PROMPT


def deterministic_faithfulness(events: list[EventRecord]) -> float:
    if not events:
        return 1.0

    avg_confidence = mean(event.confidence for event in events)
    evidence_quality = mean(1.0 if len(event.evidence_text) >= 25 else 0.6 for event in events)
    unique_reports = len({event.report_id for event in events})
    support_coverage = min(1.0, unique_reports / max(1, len(events)))

    score = 0.55 * avg_confidence + 0.25 * evidence_quality + 0.20 * support_coverage
    return max(0.0, min(1.0, score))


def llm_faithfulness(
    llm: BaseChatModel | None,
    question: str,
    answer: str,
    events: list[EventRecord],
) -> float | None:
    if llm is None:
        return None

    evidence = "\n".join(
        f"[{event.report_id} | {event.report_date}] {event.evidence_text}"
        for event in events[:25]
    )
    chain = VALIDATOR_PROMPT | llm

    try:
        raw = chain.invoke({"question": question, "answer": answer, "evidence": evidence})
    except Exception:
        return None

    content = _read_content(raw)

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None

    raw_score = payload.get("faithful_score")
    if isinstance(raw_score, (int, float)):
        return max(0.0, min(1.0, float(raw_score)))
    return None


def combine_faithfulness(base_score: float, llm_score: float | None) -> float:
    if llm_score is None:
        return base_score
    return max(0.0, min(1.0, 0.6 * base_score + 0.4 * llm_score))


def _read_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(item) for item in content).strip()
    return str(content).strip()
