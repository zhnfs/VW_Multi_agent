from acid_agent.agents.planner import PlannerAgent
from acid_agent.models import QueryIntent


def test_planner_detects_count_intent() -> None:
    planner = PlannerAgent(llm=None)
    plan = planner.plan("How many acid jobs were done for well ABC-1001?")
    assert plan.intent == QueryIntent.COUNT
    assert plan.well_id == "ABC-1001"


def test_planner_detects_subtypes_intent() -> None:
    planner = PlannerAgent(llm=None)
    plan = planner.plan("List subtypes for WELL-88")
    assert plan.intent == QueryIntent.SUBTYPES
    assert plan.well_id == "WELL-88"


def test_planner_uses_explicit_well_id() -> None:
    planner = PlannerAgent(llm=None)
    plan = planner.plan("How many acid jobs happened?", explicit_well_id="MANUAL-1")
    assert plan.well_id == "MANUAL-1"
