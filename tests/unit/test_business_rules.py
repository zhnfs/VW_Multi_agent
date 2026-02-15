from acid_agent.business_rules import infer_subtype_by_rules, is_acid_job_sentence
from acid_agent.models import AcidSubtype


def test_is_acid_job_sentence_positive() -> None:
    text = "Pumped 120 bbl 15% HCL acid treatment to stimulate the interval."
    assert is_acid_job_sentence(text)


def test_is_acid_job_sentence_negative() -> None:
    text = "Moved acid totes to location and prepared pumps for tomorrow."
    assert not is_acid_job_sentence(text)


def test_infer_subtype_fracturing() -> None:
    text = "Performed acid frac stage with high pressure fracture extension."
    assert infer_subtype_by_rules(text) == AcidSubtype.ACID_FRACTURING


def test_infer_subtype_wash() -> None:
    text = "Completed tubing acid wash to dissolve scale and clean completion."
    assert infer_subtype_by_rules(text) == AcidSubtype.ACID_WASH
