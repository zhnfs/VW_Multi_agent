from ops_assistant.agents import (
    ClassificationAgent,
    ExtractionAgent,
    MultiAgentOrchestrator,
    PlannerAgent,
    ValidationAgent,
)
from ops_assistant.data_access import InMemoryRepository
from ops_assistant.models import EventSubtype, ReportRecord


def test_end_to_end_pipeline_counts_and_subtypes() -> None:
    reports = [
        ReportRecord(
            report_id="R-001",
            asset_id="ASSET-42",
            report_date="2025-01-01",
            report_text="Pumped 120 bbl 12% treatment blend in gamma sequence.",
        ),
        ReportRecord(
            report_id="R-002",
            asset_id="ASSET-42",
            report_date="2025-01-02",
            report_text="Performed alpha stage intervention across the interval.",
        ),
        ReportRecord(
            report_id="R-003",
            asset_id="ASSET-42",
            report_date="2025-01-03",
            report_text="Executed beta treatment phase at high pressure for propagation.",
        ),
        ReportRecord(
            report_id="R-004",
            asset_id="ASSET-42",
            report_date="2025-01-04",
            report_text="Performed delta treatment pre-stage before the main operation.",
        ),
        ReportRecord(
            report_id="R-005",
            asset_id="ASSET-42",
            report_date="2025-01-05",
            report_text="Moved materials and inspected pump lines. No operation performed.",
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
        question="For asset ASSET-42, give me event count and subtype distribution",
        explicit_asset_id="ASSET-42",
    )

    assert response.total_events == 4
    assert response.subtype_counts[EventSubtype.CATEGORY_GAMMA] == 1
    assert response.subtype_counts[EventSubtype.CATEGORY_ALPHA] == 1
    assert response.subtype_counts[EventSubtype.CATEGORY_BETA] == 1
    assert response.subtype_counts[EventSubtype.CATEGORY_DELTA] == 1
    assert response.faithful_score >= 0.5
