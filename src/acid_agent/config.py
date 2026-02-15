from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    uc_catalog: str
    uc_schema: str
    uc_reports_table: str
    databricks_server_hostname: str | None
    databricks_http_path: str | None
    databricks_token: str | None
    openai_api_key: str | None
    openai_model: str
    mlflow_experiment_name: str
    min_faithful_score: float

    @property
    def fully_qualified_reports_table(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.uc_reports_table}"

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            uc_catalog=os.getenv("UC_CATALOG", "main"),
            uc_schema=os.getenv("UC_SCHEMA", "well_ops"),
            uc_reports_table=os.getenv("UC_REPORTS_TABLE", "well_daily_reports"),
            databricks_server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            databricks_http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            databricks_token=os.getenv("DATABRICKS_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/acid-job-agent"),
            min_faithful_score=float(os.getenv("MIN_FAITHFUL_SCORE", "0.80")),
        )
