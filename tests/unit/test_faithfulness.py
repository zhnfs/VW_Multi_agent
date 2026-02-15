from acid_agent.faithfulness import combine_faithfulness, deterministic_faithfulness
from acid_agent.models import AcidJobEvent, AcidSubtype


def _sample_event(confidence: float) -> AcidJobEvent:
    return AcidJobEvent(
        event_id="e1",
        well_id="W-1",
        report_id="R-1",
        report_date="2025-01-01",
        evidence_text="Pumped 100 bbl HCL acid wash treatment.",
        subtype=AcidSubtype.ACID_WASH,
        confidence=confidence,
    )


def test_deterministic_faithfulness_increases_with_confidence() -> None:
    low = deterministic_faithfulness([_sample_event(0.45)])
    high = deterministic_faithfulness([_sample_event(0.90)])
    assert high > low


def test_combine_faithfulness_uses_llm_when_present() -> None:
    combined = combine_faithfulness(0.6, 0.9)
    assert 0.6 < combined < 0.9
