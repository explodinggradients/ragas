"""tracing using mlflow"""

__all__ = ["MLflowTrace", "sync_trace"]

import os
import typing as t

from mlflow import get_last_active_trace
from mlflow.entities.span import Span
from mlflow.entities.trace import Trace


class MLflowTrace:

    def __init__(self, trace: Trace):
        self.trace = trace

    def get_url(self) -> str:

        server_url = os.getenv("MLFLOW_HOST")
        if not server_url:
            raise ValueError("MLFLOW_HOST environment variable is not set.")
        trace_info = self.trace.info
        server_url = server_url.rstrip("/")
        request_id = trace_info.request_id
        experiment_id = trace_info.experiment_id

        # Build the trace URL
        trace_url = (
            f"{server_url}/#/experiments/{experiment_id}?"
            f"compareRunsMode=TRACES&"
            f"selectedTraceId={request_id}"
        )

        return trace_url

    def get_filter(self, span_name) -> t.List[Span]:

        return self.trace.search_spans(name=span_name)


async def sync_trace():

    trace = get_last_active_trace()
    if trace is None:
        raise ValueError("No active trace found.")

    return MLflowTrace(trace)
