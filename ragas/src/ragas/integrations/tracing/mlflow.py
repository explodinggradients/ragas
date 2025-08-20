"""tracing using mlflow"""

__all__ = ["MLflowTrace", "sync_trace"]

import os
import typing as t

if t.TYPE_CHECKING:
    from mlflow import get_last_active_trace_id, get_trace
    from mlflow.entities.span import Span
    from mlflow.entities.trace import Trace
else:
    try:
        from mlflow import get_last_active_trace_id, get_trace  # type: ignore
        from mlflow.entities.span import Span  # type: ignore
        from mlflow.entities.trace import Trace  # type: ignore

        MLFLOW_AVAILABLE = True
    except ImportError:
        MLFLOW_AVAILABLE = False

        # Define stub classes for type checking when imports fail
        class Span:  # type: ignore
            name: str = ""

        class Trace:  # type: ignore
            def __init__(self):  # type: ignore
                self.info = type(
                    "TraceInfo", (), {"request_id": "", "experiment_id": ""}
                )()

            def search_spans(self, name: str) -> t.List["Span"]:  # type: ignore
                return []

        def get_last_active_trace_id() -> t.Optional[str]:  # type: ignore
            return None

        def get_trace(trace_id: str) -> t.Optional["Trace"]:  # type: ignore
            return None


class MLflowTrace:
    def __init__(self, trace: "Trace"):
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

    def get_filter(self, span_name: str) -> t.List["Span"]:
        return self.trace.search_spans(name=span_name)


async def sync_trace() -> MLflowTrace:
    trace_id = get_last_active_trace_id()
    if trace_id is None:
        raise ValueError("No active trace found.")

    trace = get_trace(trace_id)
    if trace is None:
        raise ValueError("Trace not found.")

    return MLflowTrace(trace)
