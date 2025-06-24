"""Utils to help to interact with langfuse traces"""

__all__ = ["observe", "logger", "LangfuseTrace", "sync_trace", "add_query_param"]

import asyncio
import logging
import typing as t
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from langfuse.api import Observation, TraceWithFullDetails
from langfuse.decorators import langfuse_context, observe
from langfuse.utils.langfuse_singleton import LangfuseSingleton

# just adding it to the namespace
observe = observe

logger = logging.getLogger(__name__)


class LangfuseTrace:
    def __init__(self, trace: TraceWithFullDetails):
        self.trace = trace
        self._langfuse_client = langfuse_context.client_instance

    def get_url(self):
        return langfuse_context.get_current_trace_url()

    def filter(self, span_name: str) -> t.List[Observation]:
        trace = self._langfuse_client.fetch_trace(self.trace.id)
        return [span for span in trace.data.observations if span.name == span_name]


async def sync_trace(
    trace_id: t.Optional[str] = None, max_retries: int = 10, delay: float = 2
) -> LangfuseTrace:
    """Wait for a Langfuse trace to be synced to the server.

    Args:
        trace_id: The ID of the trace to wait for
        max_retries: Maximum number of retry attempts (default: 10)
        delay: Delay between retries in seconds (default: 0.5)

    Returns:
        Trace object if found, None if not found after retries
    """
    if trace_id is None:
        # if no trace id is provided, get the current trace id
        trace_id = langfuse_context.get_current_trace_id()

    if not trace_id:
        raise ValueError(
            "No trace id found. Please ensure you are running this function within a function decorated with @observe()."
        )
    for _ in range(max_retries):
        langfuse_client = LangfuseSingleton().get()
        try:
            # you can also use the async api - langfuse_client.async_api.trace.get(trace_id)
            # .client might be deprecated in the future (didn't change it for superme)
            trace = langfuse_client.client.trace.get(trace_id)
            if trace:
                return LangfuseTrace(trace=trace)
        except Exception as e:
            logger.debug(f"Trace {trace_id} not yet synced: {str(e)}")

        await asyncio.sleep(delay)

    raise ValueError(f"Trace {trace_id} not found after {max_retries} attempts")


def add_query_param(url, param_name, param_value):
    """Add a query parameter to a URL."""
    # Parse the URL
    url_parts = list(urlparse(url))

    # Get query params as a dict and add new param
    query_dict = dict(parse_qsl(url_parts[4]))
    query_dict[param_name] = param_value

    # Replace the query part with updated params
    url_parts[4] = urlencode(query_dict)

    # Reconstruct the URL
    return urlunparse(url_parts)
