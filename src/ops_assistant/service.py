from __future__ import annotations

from ops_assistant.agents import (
    ClassificationAgent,
    ExtractionAgent,
    MultiAgentOrchestrator,
    PlannerAgent,
    ValidationAgent,
)
from ops_assistant.config import AppConfig
from ops_assistant.data_access import ReportRepository, SqlWarehouseRepository
from ops_assistant.llm import build_llm
from ops_assistant.mlflow_utils import initialize_mlflow


def build_orchestrator(
    config: AppConfig | None = None,
    repository_override: ReportRepository | None = None,
) -> MultiAgentOrchestrator:
    app_config = config or AppConfig.from_env()
    initialize_mlflow(app_config.mlflow_experiment_name)

    llm = build_llm(app_config)
    repository = repository_override or SqlWarehouseRepository(app_config)

    return MultiAgentOrchestrator(
        repository=repository,
        planner=PlannerAgent(llm=llm),
        extractor=ExtractionAgent(),
        classifier=ClassificationAgent(llm=llm),
        validator=ValidationAgent(min_faithful_score=app_config.min_faithful_score, llm=llm),
    )
