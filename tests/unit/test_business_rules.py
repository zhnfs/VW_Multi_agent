from ops_assistant.business_rules import infer_category_by_rules, is_target_event_sentence
from ops_assistant.models import EventSubtype


def test_is_target_event_sentence_positive() -> None:
    text = "Pumped 120 bbl 12% treatment blend in phase A operation."
    assert is_target_event_sentence(text)


def test_is_target_event_sentence_negative() -> None:
    text = "Moved treatment totes to location and prepared pumps for tomorrow."
    assert not is_target_event_sentence(text)


def test_infer_subtype_beta() -> None:
    text = "Performed beta stage with high pressure execution details."
    assert infer_category_by_rules(text) == EventSubtype.CATEGORY_BETA


def test_infer_subtype_gamma() -> None:
    text = "Completed gamma sequence to circulate and clean the interval."
    assert infer_category_by_rules(text) == EventSubtype.CATEGORY_GAMMA
