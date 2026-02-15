from __future__ import annotations

import mlflow
import pytest


@pytest.fixture(autouse=True)
def isolate_mlflow_tracking(tmp_path_factory: pytest.TempPathFactory) -> None:
    old_uri = mlflow.get_tracking_uri()
    tracking_dir = tmp_path_factory.mktemp("mlflow-tracking")
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("test-experiment")

    yield

    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)
