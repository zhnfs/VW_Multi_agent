from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress

import mlflow


def initialize_mlflow(experiment_name: str) -> None:
    with suppress(Exception):
        mlflow.set_experiment(experiment_name)

    try:
        mlflow.langchain.autolog(log_traces=True)
    except TypeError:
        mlflow.langchain.autolog()
    except Exception:
        pass


@contextmanager
def run_context(run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    active = mlflow.active_run()
    if active:
        if tags:
            mlflow.set_tags(tags)
        yield
        return

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        yield


@contextmanager
def traced_span(name: str, attributes: dict[str, str] | None = None) -> Iterator[None]:
    if hasattr(mlflow, "start_span"):
        payload = {k: str(v) for k, v in (attributes or {}).items()}
        with mlflow.start_span(name=name, attributes=payload):
            yield
        return

    active = mlflow.active_run()
    with mlflow.start_run(run_name=name, nested=active is not None):
        for key, value in (attributes or {}).items():
            mlflow.log_param(f"span.{name}.{key}", str(value))
        yield
