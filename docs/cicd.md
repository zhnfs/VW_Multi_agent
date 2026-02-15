# CI/CD Setup Guide (GitHub + Databricks Apps)

## What Is Included
- CI workflow: `.github/workflows/ci.yml`
  - Installs dependencies with `uv`
  - Runs lint + tests
- CD workflow: `.github/workflows/cd-databricks.yml`
  - Authenticates Databricks CLI
  - Creates app if missing
  - Deploys source code to Databricks App

## Required GitHub Secrets
Add these in GitHub repo settings under **Secrets and variables > Actions**:
- `DATABRICKS_HOST` (example: `https://adb-xxxx.azuredatabricks.net`)
- `DATABRICKS_TOKEN` (PAT or service-principal token)
- `DATABRICKS_APP_NAME` (target Databricks App name)

## Recommended GitHub Protections
1. Protect `main` branch.
2. Require CI workflow (`ci`) to pass before merge.
3. Require pull request reviews.

## Local Validation Before Pushing
```bash
uv sync --group dev
uv run ruff check .
uv run pytest
```

## Manual Deployment (Fallback)
If GitHub Actions is not available, deploy manually:
```bash
export DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
export DATABRICKS_TOKEN=***

# Create once if needed
databricks apps create <app-name>

# Deploy current directory
databricks apps deploy <app-name> --source-code-path .
```

## Production Hardening Checklist
1. Replace placeholder SME prompt in `src/acid_agent/resources/sme_rules_placeholder.txt`.
2. Add a labeled benchmark suite and fail CI if accuracy drops below 95%.
3. Rotate Databricks tokens to service principal auth.
4. Add environment-specific workflows (dev/staging/prod) with approvals.
