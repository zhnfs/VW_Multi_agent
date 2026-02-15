from __future__ import annotations

from acid_agent.agents import (
    ClassificationAgent,
    ExtractionAgent,
    MultiAgentOrchestrator,
    PlannerAgent,
    ValidationAgent,
)
from acid_agent.config import AppConfig
from acid_agent.data_access import DatabricksUCRepository, WellReportRepository
from acid_agent.llm import build_llm
from acid_agent.mlflow_utils import initialize_mlflow


def build_orchestrator(
    config: AppConfig | None = None,
    repository_override: WellReportRepository | None = None,
) -> MultiAgentOrchestrator:
    app_config = config or AppConfig.from_env()
    initialize_mlflow(app_config.mlflow_experiment_name)

    llm = build_llm(app_config)
    repository = repository_override or DatabricksUCRepository(app_config)

    return MultiAgentOrchestrator(
        repository=repository,
        planner=PlannerAgent(llm=llm),
        extractor=ExtractionAgent(),
        classifier=ClassificationAgent(llm=llm),
        validator=ValidationAgent(min_faithful_score=app_config.min_faithful_score, llm=llm),
    )
