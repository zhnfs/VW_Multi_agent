from __future__ import annotations

from collections import Counter

from langchain_core.language_models.chat_models import BaseChatModel

from acid_agent.faithfulness import (
    combine_faithfulness,
    deterministic_faithfulness,
    llm_faithfulness,
)
from acid_agent.models import AcidJobEvent, AcidSubtype, QueryIntent


class ValidationAgent:
    def __init__(self, min_faithful_score: float, llm: BaseChatModel | None = None) -> None:
        self.min_faithful_score = min_faithful_score
        self.llm = llm

    def validate(
        self,
        question: str,
        well_id: str,
        intent: QueryIntent,
        events: list[AcidJobEvent],
    ) -> tuple[float, dict[AcidSubtype, int], list[str]]:
        subtype_counter: Counter[AcidSubtype] = Counter(
            event.subtype for event in events if event.subtype is not None
        )
        subtype_counts: dict[AcidSubtype, int] = {
            subtype: subtype_counter.get(subtype, 0) for subtype in AcidSubtype
        }

        provisional_answer = _build_provisional_answer(
            intent=intent,
            well_id=well_id,
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
            warnings.append("No acid-job evidence found in available reports for this well.")

        return faithful_score, subtype_counts, warnings


def _build_provisional_answer(
    intent: QueryIntent,
    well_id: str,
    subtype_counts: dict[AcidSubtype, int],
    events: list[AcidJobEvent],
) -> str:
    total_jobs = len(events)
    subtype_summary = ", ".join(
        f"{subtype.value}:{count}" for subtype, count in subtype_counts.items()
    )

    if intent == QueryIntent.COUNT:
        return f"Well {well_id} has {total_jobs} acid jobs."
    if intent == QueryIntent.SUBTYPES:
        return f"Well {well_id} subtype distribution is {subtype_summary}."
    return f"Well {well_id} has {total_jobs} acid jobs. Subtype distribution is {subtype_summary}."
