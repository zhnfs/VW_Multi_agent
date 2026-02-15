from ops_assistant.faithfulness import combine_faithfulness, deterministic_faithfulness
from ops_assistant.models import EventRecord, EventSubtype


def _sample_event(confidence: float) -> EventRecord:
    return EventRecord(
        event_id="e1",
        asset_id="A-1",
        report_id="R-1",
        report_date="2025-01-01",
        evidence_text="Pumped 100 bbl treatment blend in gamma sequence.",
        subtype=EventSubtype.CATEGORY_GAMMA,
        confidence=confidence,
    )


def test_deterministic_faithfulness_increases_with_confidence() -> None:
    low = deterministic_faithfulness([_sample_event(0.45)])
    high = deterministic_faithfulness([_sample_event(0.90)])
    assert high > low


def test_combine_faithfulness_uses_llm_when_present() -> None:
    combined = combine_faithfulness(0.6, 0.9)
    assert 0.6 < combined < 0.9
