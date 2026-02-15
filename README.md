# Event Insight Multi-Agent System

Streamlit + LangChain multi-agent application to answer:
1. How many target events were performed for a specific asset.
2. What event subtypes were performed (4 subtype taxonomy).

## Scope and Constraints
- Data source: Databricks Unity Catalog table with daily reports (many reports per asset).
- Required targets: 95% accuracy for count and 95% for subtype classification.
- Faithfulness score included in every response to reduce hallucination risk.
- MLflow traces/logs at each critical stage.

## Subtype Taxonomy (4)
- `category_alpha`
- `category_beta`
- `category_gamma`
- `category_delta`

## Architecture
- `PlannerAgent`: detects intent and asset id.
- `ExtractionAgent`: finds target-event evidence from report text using rules.
- `ClassificationAgent`: maps evidence to one of the 4 subtypes (rules first, LLM fallback).
- `ValidationAgent`: computes faithfulness score and warnings.
- `MultiAgentOrchestrator`: coordinates all agents and logs traces/metrics in MLflow.

## Project Layout
- `app.py`: Streamlit app entrypoint.
- `src/ops_assistant/app.py`: Chatbot UI (preset and free-form questions).
- `src/ops_assistant/data_access.py`: Databricks SQL report fetcher + in-memory repository for tests.
- `src/ops_assistant/agents/`: multi-agent pipeline.
- `src/ops_assistant/resources/sme_rules_placeholder.txt`: placeholder prompt/rules.
- `tests/unit/`: unit tests.
- `tests/integration/`: integration tests.
- `.github/workflows/`: CI and Databricks CD workflows.

## Required Source Schema
The query expects a table with these columns:
- `report_id` (string)
- `asset_id` (string)
- `report_date` (date or timestamp)
- `report_text` (string)

Set env vars to point to your table:
- `SOURCE_CATALOG`
- `SOURCE_SCHEMA`
- `SOURCE_REPORTS_TABLE`
- `SOURCE_ASSET_ID_COLUMN`

## Local Setup (uv)
```bash
uv sync --group dev
```

Run app:
```bash
uv run streamlit run app.py
```

## Environment Variables
```bash
export SOURCE_CATALOG=main
export SOURCE_SCHEMA=ops_data
export SOURCE_REPORTS_TABLE=daily_reports
export SOURCE_ASSET_ID_COLUMN=asset_id

export DATABRICKS_SERVER_HOSTNAME=adb-xxxx.xx.azuredatabricks.net
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxx
export DATABRICKS_TOKEN=...

export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini

export MLFLOW_EXPERIMENT_NAME=/Shared/event-insight
export MIN_FAITHFUL_SCORE=0.80
```

## Testing
Run all tests:
```bash
uv run python -m pytest
```

## Accuracy Validation Path
Use `src/ops_assistant/evaluation.py` with a labeled benchmark set of assets:
- `count_accuracy >= 0.95`
- `subtype_accuracy >= 0.95`

Gate release in CI/CD only when both thresholds pass.

## CI/CD
- CI workflow: `.github/workflows/ci.yml`
- CD workflow: `.github/workflows/cd-app.yml`

See `docs/cicd.md` for required secrets and deployment steps.
