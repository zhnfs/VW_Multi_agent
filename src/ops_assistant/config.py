from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    source_catalog: str
    source_schema: str
    source_reports_table: str
    source_asset_id_column: str
    databricks_server_hostname: str | None
    databricks_http_path: str | None
    databricks_token: str | None
    openai_api_key: str | None
    openai_model: str
    mlflow_experiment_name: str
    min_faithful_score: float

    @property
    def fully_qualified_reports_table(self) -> str:
        return f"{self.source_catalog}.{self.source_schema}.{self.source_reports_table}"

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            source_catalog=os.getenv("SOURCE_CATALOG") or os.getenv("UC_CATALOG", "main"),
            source_schema=os.getenv("SOURCE_SCHEMA") or os.getenv("UC_SCHEMA", "ops_data"),
            source_reports_table=os.getenv("SOURCE_REPORTS_TABLE")
            or os.getenv("UC_REPORTS_TABLE", "daily_reports"),
            source_asset_id_column=os.getenv("SOURCE_ASSET_ID_COLUMN", "asset_id"),
            databricks_server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            databricks_http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            databricks_token=os.getenv("DATABRICKS_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/event-insight"),
            min_faithful_score=float(os.getenv("MIN_FAITHFUL_SCORE", "0.80")),
        )
