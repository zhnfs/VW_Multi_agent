from __future__ import annotations

from dataclasses import dataclass

from acid_agent.agents.orchestrator import MultiAgentOrchestrator
from acid_agent.models import AcidSubtype


@dataclass(frozen=True)
class LabeledCase:
    question: str
    well_id: str
    expected_count: int
    expected_subtypes: dict[AcidSubtype, int]


@dataclass(frozen=True)
class EvaluationResult:
    count_accuracy: float
    subtype_accuracy: float
    meets_count_target: bool
    meets_subtype_target: bool


def evaluate_accuracy(
    orchestrator: MultiAgentOrchestrator,
    cases: list[LabeledCase],
    count_target: float = 0.95,
    subtype_target: float = 0.95,
) -> EvaluationResult:
    if not cases:
        raise ValueError("At least one labeled case is required for evaluation.")

    count_hits = 0
    subtype_hits = 0

    for case in cases:
        response = orchestrator.answer(question=case.question, explicit_well_id=case.well_id)
        if response.total_acid_jobs == case.expected_count:
            count_hits += 1
        if response.subtype_counts == case.expected_subtypes:
            subtype_hits += 1

    count_accuracy = count_hits / len(cases)
    subtype_accuracy = subtype_hits / len(cases)

    return EvaluationResult(
        count_accuracy=count_accuracy,
        subtype_accuracy=subtype_accuracy,
        meets_count_target=count_accuracy >= count_target,
        meets_subtype_target=subtype_accuracy >= subtype_target,
    )
