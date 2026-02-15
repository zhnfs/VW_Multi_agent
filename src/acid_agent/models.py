from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    COUNT = "count"
    SUBTYPES = "subtypes"
    BOTH = "both"


class AcidSubtype(StrEnum):
    MATRIX_ACIDIZING = "matrix_acidizing"
    ACID_FRACTURING = "acid_fracturing"
    ACID_WASH = "acid_wash"
    ACID_SPEARHEAD = "acid_spearhead"


class ReportRecord(BaseModel):
    report_id: str
    well_id: str
    report_date: str
    report_text: str


class AcidJobEvent(BaseModel):
    event_id: str
    well_id: str
    report_id: str
    report_date: str
    evidence_text: str
    subtype: AcidSubtype | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentPlan(BaseModel):
    intent: QueryIntent
    well_id: str


class AgentResponse(BaseModel):
    question: str
    well_id: str
    intent: QueryIntent
    total_acid_jobs: int
    subtype_counts: dict[AcidSubtype, int]
    events: list[AcidJobEvent]
    faithful_score: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "well_id": self.well_id,
            "intent": self.intent.value,
            "total_acid_jobs": self.total_acid_jobs,
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
            answer = f"Detected {self.total_acid_jobs} acid job(s) for well `{self.well_id}`."
        elif self.intent == QueryIntent.SUBTYPES:
            answer = (
                f"Detected acid-job subtype distribution for well `{self.well_id}`:\n"
                f"{subtype_section}"
            )
        else:
            answer = (
                f"Detected {self.total_acid_jobs} acid job(s) for well `{self.well_id}` "
                "with subtype distribution:\n"
                f"{subtype_section}"
            )

        answer += f"\n\nFaithfulness score: {self.faithful_score:.2%}."
        if self.warnings:
            answer += "\nWarnings:\n" + "\n".join(f"- {item}" for item in self.warnings)
        return answer
