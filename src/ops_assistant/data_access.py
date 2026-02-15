from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ops_assistant.config import AppConfig
from ops_assistant.models import ReportRecord


class ReportRepository(Protocol):
    def fetch_reports(self, asset_id: str) -> list[ReportRecord]:
        ...


@dataclass
class InMemoryRepository(ReportRepository):
    reports: list[ReportRecord]

    def fetch_reports(self, asset_id: str) -> list[ReportRecord]:
        return [report for report in self.reports if report.asset_id == asset_id]


class SqlWarehouseRepository(ReportRepository):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def fetch_reports(self, asset_id: str) -> list[ReportRecord]:
        if not (
            self.config.databricks_server_hostname
            and self.config.databricks_http_path
            and self.config.databricks_token
        ):
            raise ValueError(
                "Missing Databricks SQL credentials. Configure DATABRICKS_SERVER_HOSTNAME, "
                "DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN."
            )

        try:
            from databricks import sql
        except ImportError as exc:
            raise ImportError("databricks-sql-connector is required") from exc

        safe_asset_id = asset_id.replace("'", "''")
        id_column = self.config.source_asset_id_column
        query = (
            f"SELECT report_id, {id_column} AS asset_id, "
            "CAST(report_date AS STRING) AS report_date, report_text "
            f"FROM {self.config.fully_qualified_reports_table} "
            f"WHERE {id_column} = '{safe_asset_id}' "
            "ORDER BY report_date"
        )

        with sql.connect(
            server_hostname=self.config.databricks_server_hostname,
            http_path=self.config.databricks_http_path,
            access_token=self.config.databricks_token,
        ) as connection, connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        return [
            ReportRecord(
                report_id=str(row[0]),
                asset_id=str(row[1]),
                report_date=str(row[2]),
                report_text=str(row[3]),
            )
            for row in rows
        ]
