from __future__ import annotations

import mlflow

from acid_agent.agents.classifier import ClassificationAgent
from acid_agent.agents.extractor import ExtractionAgent
from acid_agent.agents.planner import PlannerAgent
from acid_agent.agents.validator import ValidationAgent
from acid_agent.data_access import WellReportRepository
from acid_agent.mlflow_utils import run_context, traced_span
from acid_agent.models import AgentResponse


class MultiAgentOrchestrator:
    def __init__(
        self,
        repository: WellReportRepository,
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

    def answer(self, question: str, explicit_well_id: str | None = None) -> AgentResponse:
        run_name = f"acid-job-query-{(explicit_well_id or 'unknown').lower()}"
        with run_context(run_name=run_name, tags={"app": "acid-job-agent"}):
            with traced_span("planning", {"question": question}):
                plan = self.planner.plan(question=question, explicit_well_id=explicit_well_id)

            with traced_span("fetch_reports", {"well_id": plan.well_id}):
                reports = self.repository.fetch_reports(plan.well_id)

            with traced_span("extract_jobs", {"report_count": str(len(reports))}):
                events = self.extractor.extract(reports)

            with traced_span("classify_subtypes", {"event_count": str(len(events))}):
                classified_events = self.classifier.classify(events)

            with traced_span("validate", {"event_count": str(len(classified_events))}):
                faithful_score, subtype_counts, warnings = self.validator.validate(
                    question=question,
                    well_id=plan.well_id,
                    intent=plan.intent,
                    events=classified_events,
                )

            response = AgentResponse(
                question=question,
                well_id=plan.well_id,
                intent=plan.intent,
                total_acid_jobs=len(classified_events),
                subtype_counts=subtype_counts,
                events=classified_events,
                faithful_score=faithful_score,
                warnings=warnings,
            )

            self._log_response_metrics(response)
            return response

    @staticmethod
    def _log_response_metrics(response: AgentResponse) -> None:
        mlflow.log_metric("total_acid_jobs", float(response.total_acid_jobs))
        mlflow.log_metric("faithful_score", response.faithful_score)
        for subtype, count in response.subtype_counts.items():
            mlflow.log_metric(f"subtype_count_{subtype.value}", float(count))
        if response.warnings:
            mlflow.log_param("warnings", " | ".join(response.warnings))
        mlflow.log_dict(response.to_dict(), "response.json")
