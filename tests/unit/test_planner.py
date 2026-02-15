from ops_assistant.agents.planner import PlannerAgent
from ops_assistant.models import QueryIntent


def test_planner_detects_count_intent() -> None:
    planner = PlannerAgent(llm=None)
    plan = planner.plan("How many target events were done for asset ABC-1001?")
    assert plan.intent == QueryIntent.COUNT
    assert plan.asset_id == "ABC-1001"


def test_planner_detects_subtypes_intent() -> None:
    planner = PlannerAgent(llm=None)
    plan = planner.plan("List subtypes for asset AST-88")
    assert plan.intent == QueryIntent.SUBTYPES
    assert plan.asset_id == "AST-88"


def test_planner_uses_explicit_asset_id() -> None:
    planner = PlannerAgent(llm=None)
    plan = planner.plan("How many target events happened?", explicit_asset_id="MANUAL-1")
    assert plan.asset_id == "MANUAL-1"
