from __future__ import annotations

import mlflow

from ops_assistant.agents.classifier import ClassificationAgent
from ops_assistant.agents.extractor import ExtractionAgent
from ops_assistant.agents.planner import PlannerAgent
from ops_assistant.agents.validator import ValidationAgent
from ops_assistant.data_access import ReportRepository
from ops_assistant.mlflow_utils import run_context, traced_span
from ops_assistant.models import AgentResponse


class MultiAgentOrchestrator:
    def __init__(
        self,
        repository: ReportRepository,
        planner: PlannerAgent,
        extractor: ExtractionAgent,
        classifier: ClassificationAgent,
        validator: ValidationAgent,
    ) -> None:
        self.repository = repository
        self.planner = planner
        self.extractor = extractor
        self.classifier = classifier
        self.validator = validator

    def answer(self, question: str, explicit_asset_id: str | None = None) -> AgentResponse:
        run_name = f"event-query-{(explicit_asset_id or 'unknown').lower()}"
        with run_context(run_name=run_name, tags={"app": "ops-assistant"}):
            with traced_span("planning", {"question": question}):
                plan = self.planner.plan(question=question, explicit_asset_id=explicit_asset_id)

            with traced_span("fetch_reports", {"asset_id": plan.asset_id}):
                reports = self.repository.fetch_reports(plan.asset_id)

            with traced_span("extract_events", {"report_count": str(len(reports))}):
                events = self.extractor.extract(reports)

            with traced_span("classify_subtypes", {"event_count": str(len(events))}):
                classified_events = self.classifier.classify(events)

            with traced_span("validate", {"event_count": str(len(classified_events))}):
                faithful_score, subtype_counts, warnings = self.validator.validate(
                    question=question,
                    asset_id=plan.asset_id,
                    intent=plan.intent,
                    events=classified_events,
                )

            response = AgentResponse(
                question=question,
                asset_id=plan.asset_id,
                intent=plan.intent,
                total_events=len(classified_events),
                subtype_counts=subtype_counts,
                events=classified_events,
                faithful_score=faithful_score,
                warnings=warnings,
            )

            self._log_response_metrics(response)
            return response

    @staticmethod
    def _log_response_metrics(response: AgentResponse) -> None:
        mlflow.log_metric("total_events", float(response.total_events))
        mlflow.log_metric("faithful_score", response.faithful_score)
        for subtype, count in response.subtype_counts.items():
            mlflow.log_metric(f"subtype_count_{subtype.value}", float(count))
        if response.warnings:
            mlflow.log_param("warnings", " | ".join(response.warnings))
        mlflow.log_dict(response.to_dict(), "response.json")
