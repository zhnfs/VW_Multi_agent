from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    COUNT = "count"
    SUBTYPES = "subtypes"
    BOTH = "both"


class EventSubtype(StrEnum):
    CATEGORY_ALPHA = "category_alpha"
    CATEGORY_BETA = "category_beta"
    CATEGORY_GAMMA = "category_gamma"
    CATEGORY_DELTA = "category_delta"


class ReportRecord(BaseModel):
    report_id: str
    asset_id: str
    report_date: str
    report_text: str


class EventRecord(BaseModel):
    event_id: str
    asset_id: str
    report_id: str
    report_date: str
    evidence_text: str
    subtype: EventSubtype | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentPlan(BaseModel):
    intent: QueryIntent
    asset_id: str


class AgentResponse(BaseModel):
    question: str
    asset_id: str
    intent: QueryIntent
    total_events: int
    subtype_counts: dict[EventSubtype, int]
    events: list[EventRecord]
    faithful_score: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "asset_id": self.asset_id,
            "intent": self.intent.value,
            "total_events": self.total_events,
            "subtype_counts": {k.value: v for k, v in self.subtype_counts.items()},
            "faithful_score": round(self.faithful_score, 4),
            "warnings": self.warnings,
            "events": [event.model_dump(mode="json") for event in self.events],
        }

    def render_answer(self) -> str:
        subtype_lines = []
        for subtype, count in sorted(self.subtype_counts.items(), key=lambda item: item[0].value):
            subtype_lines.append(f"- {subtype.value}: {count}")
        subtype_section = "\n".join(subtype_lines) if subtype_lines else "- no subtypes found"

        if self.intent == QueryIntent.COUNT:
            answer = f"Detected {self.total_events} target event(s) for asset `{self.asset_id}`."
        elif self.intent == QueryIntent.SUBTYPES:
            answer = (
                f"Detected subtype distribution for asset `{self.asset_id}`:\n"
                f"{subtype_section}"
            )
        else:
            answer = (
                f"Detected {self.total_events} target event(s) for asset `{self.asset_id}` "
                "with subtype distribution:\n"
                f"{subtype_section}"
            )

        answer += f"\n\nFaithfulness score: {self.faithful_score:.2%}."
        if self.warnings:
            answer += "\nWarnings:\n" + "\n".join(f"- {item}" for item in self.warnings)
        return answer
