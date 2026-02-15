from acid_agent.agents import (
    ClassificationAgent,
    ExtractionAgent,
    MultiAgentOrchestrator,
    PlannerAgent,
    ValidationAgent,
)
from acid_agent.data_access import InMemoryRepository
from acid_agent.models import AcidSubtype, ReportRecord


def test_end_to_end_pipeline_counts_and_subtypes() -> None:
    reports = [
        ReportRecord(
            report_id="R-001",
            well_id="WELL-42",
            report_date="2025-01-01",
            report_text="Pumped 120 bbl 15% HCL acid wash through tubing to dissolve scale.",
        ),
        ReportRecord(
            report_id="R-002",
            well_id="WELL-42",
            report_date="2025-01-02",
            report_text="Performed matrix acidizing stimulation across perforations.",
        ),
        ReportRecord(
            report_id="R-003",
            well_id="WELL-42",
            report_date="2025-01-03",
            report_text="Executed acid frac stage at high pressure for fracture propagation.",
        ),
        ReportRecord(
            report_id="R-004",
            well_id="WELL-42",
            report_date="2025-01-04",
            report_text="Spot acid spearhead preflush before main treatment and displacement.",
        ),
        ReportRecord(
            report_id="R-005",
            well_id="WELL-42",
            report_date="2025-01-05",
            report_text="Moved chemicals and inspected pump lines. No treatment performed.",
        ),
    ]

    orchestrator = MultiAgentOrchestrator(
        repository=InMemoryRepository(reports=reports),
        planner=PlannerAgent(llm=None),
        extractor=ExtractionAgent(min_confidence=0.30),
        classifier=ClassificationAgent(llm=None),
        validator=ValidationAgent(min_faithful_score=0.5, llm=None),
    )

    response = orchestrator.answer(
        question="For well WELL-42, give me acid job count and subtype distribution",
        explicit_well_id="WELL-42",
    )

    assert response.total_acid_jobs == 4
    assert response.subtype_counts[AcidSubtype.ACID_WASH] == 1
    assert response.subtype_counts[AcidSubtype.MATRIX_ACIDIZING] == 1
    assert response.subtype_counts[AcidSubtype.ACID_FRACTURING] == 1
    assert response.subtype_counts[AcidSubtype.ACID_SPEARHEAD] == 1
    assert response.faithful_score >= 0.5
