# Acid Job Multi-Agent System

Databricks Streamlit + LangChain multi-agent application to answer:
1. How many acid jobs were performed for a specific well.
2. What acid-job subtypes were performed (4 subtype taxonomy).

## Scope and Constraints
- Data source: Unity Catalog table with well daily reports (many reports per well).
- Required targets: 95% accuracy for count and 95% for subtype classification.
- Faithfulness score included in every response to reduce hallucination risk.
- MLflow traces/logs at each critical stage.

## Subtype Taxonomy (4)
- `matrix_acidizing`
- `acid_fracturing`
- `acid_wash`
- `acid_spearhead`

## Architecture
- `PlannerAgent`: detects intent and well id.
- `ExtractionAgent`: finds acid-job evidence from report text using rules.
- `ClassificationAgent`: maps evidence to one of the 4 subtypes (rules first, LLM fallback).
- `ValidationAgent`: computes faithfulness score and warnings.
- `MultiAgentOrchestrator`: coordinates all agents and logs traces/metrics in MLflow.

## Project Layout
- `app.py`: Streamlit app entrypoint.
- `src/acid_agent/app.py`: Chatbot UI (preset and free-form questions).
- `src/acid_agent/data_access.py`: Unity Catalog report fetcher + in-memory repository for tests.
- `src/acid_agent/agents/`: multi-agent pipeline.
- `src/acid_agent/resources/sme_rules_placeholder.txt`: 500-line placeholder prompt/rules.
- `tests/unit/`: unit tests.
- `tests/integration/`: integration tests.
- `.github/workflows/`: CI and Databricks CD workflows.

## Required Unity Catalog Schema
The query expects a table with these columns:
- `report_id` (string)
- `well_id` (string)
- `report_date` (date or timestamp)
- `report_text` (string)

Set env vars to point to your table:
- `UC_CATALOG`
- `UC_SCHEMA`
- `UC_REPORTS_TABLE`

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
export UC_CATALOG=main
export UC_SCHEMA=well_ops
export UC_REPORTS_TABLE=well_daily_reports

export DATABRICKS_SERVER_HOSTNAME=adb-xxxx.xx.azuredatabricks.net
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxx
export DATABRICKS_TOKEN=...

export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini

export MLFLOW_EXPERIMENT_NAME=/Shared/acid-job-agent
export MIN_FAITHFUL_SCORE=0.80
```

## Testing
Run all tests:
```bash
uv run pytest
```

## Accuracy Validation Path
Use `src/acid_agent/evaluation.py` with a labeled benchmark set of wells:
- `count_accuracy >= 0.95`
- `subtype_accuracy >= 0.95`

Gate release in CI/CD only when both thresholds pass.

## CI/CD
- CI workflow: `.github/workflows/ci.yml`
- Databricks CD workflow: `.github/workflows/cd-databricks.yml`

See `docs/cicd.md` for required GitHub secrets and deployment steps.
