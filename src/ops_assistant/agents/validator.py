from __future__ import annotations

from collections import Counter

from langchain_core.language_models.chat_models import BaseChatModel

from ops_assistant.faithfulness import (
    combine_faithfulness,
    deterministic_faithfulness,
    llm_faithfulness,
)
from ops_assistant.models import EventRecord, EventSubtype, QueryIntent


class ValidationAgent:
    def __init__(self, min_faithful_score: float, llm: BaseChatModel | None = None) -> None:
        self.min_faithful_score = min_faithful_score
        self.llm = llm

    def validate(
        self,
        question: str,
        asset_id: str,
        intent: QueryIntent,
        events: list[EventRecord],
    ) -> tuple[float, dict[EventSubtype, int], list[str]]:
        subtype_counter: Counter[EventSubtype] = Counter(
            event.subtype for event in events if event.subtype is not None
        )
        subtype_counts: dict[EventSubtype, int] = {
            subtype: subtype_counter.get(subtype, 0) for subtype in EventSubtype
        }

        provisional_answer = _build_provisional_answer(
            intent=intent,
            asset_id=asset_id,
            subtype_counts=subtype_counts,
            events=events,
        )
        base_score = deterministic_faithfulness(events)
        judge_score = llm_faithfulness(self.llm, question, provisional_answer, events)
        faithful_score = combine_faithfulness(base_score, judge_score)

        warnings: list[str] = []
        if faithful_score < self.min_faithful_score:
            warnings.append(
                "Faithfulness below threshold. "
                "Consider manual review with full report context before operational use."
            )
        if not events:
            warnings.append("No target-event evidence found in available reports for this asset.")

        return faithful_score, subtype_counts, warnings


def _build_provisional_answer(
    intent: QueryIntent,
    asset_id: str,
    subtype_counts: dict[EventSubtype, int],
    events: list[EventRecord],
) -> str:
    total_jobs = len(events)
    subtype_summary = ", ".join(
        f"{subtype.value}:{count}" for subtype, count in subtype_counts.items()
    )

    if intent == QueryIntent.COUNT:
        return f"Asset {asset_id} has {total_jobs} target events."
    if intent == QueryIntent.SUBTYPES:
        return f"Asset {asset_id} subtype distribution is {subtype_summary}."
    return (
        f"Asset {asset_id} has {total_jobs} target events. "
        f"Subtype distribution is {subtype_summary}."
    )
