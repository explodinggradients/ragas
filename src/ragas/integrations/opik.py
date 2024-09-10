import typing as t

try:
    from opik.integrations.langchain import (  # type: ignore
        OpikTracer as LangchainOpikTracer,
    )

    from ragas.evaluation import RAGAS_EVALUATION_CHAIN_NAME
except ImportError:
    raise ImportError(
        "Opik is not installed. Please install it using `pip install opik` to use the Opik tracer."
    )

if t.TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run


class OpikTracer(LangchainOpikTracer):
    """
    Callback for Opik that can be used to log traces and evaluation scores to the Opik platform.

    Attributes
    ----------
    tags: list[string]
        The tags to set on each trace.
    metadata: dict
        Additional metadata to log for each trace.
    """

    _evaluation_run_id: t.Optional[str] = None

    def _persist_run(self, run: "Run"):
        # The _persist_run function is called by LangChain if it is a root run,
        # we update it so that we don't log the root run if we are running an evaluation.
        if run.id != self._evaluation_run_id:
            super()._persist_run(run)

    def _on_chain_start(self, run: "Run"):
        if (run.parent_run_id is None) and (run.name == RAGAS_EVALUATION_CHAIN_NAME):
            # Store the evaluation run id so we can flag the child traces and log them independently
            self._evaluation_run_id = str(run.id)
        else:
            # Each child trace of the "ragas evaluation" chain should be a new trace
            if run.parent_run_id == self._evaluation_run_id:
                run.parent_run_id = None

            super()._on_chain_start(run)

    def _on_chain_end(self, run: "Run"):
        if run.id == self._evaluation_run_id:
            pass
        else:
            # We want to log the output row chain as feedback scores as these align with the Opik terminology of "feedback scores"
            if run.name.startswith("row ") and (self._evaluation_run_id is not None):
                span = self._span_map[run.id]
                trace_id = span.trace_id
                if run.outputs:
                    self._opik_client.log_traces_feedback_scores(
                        [
                            {"id": trace_id, "name": name, "value": round(value, 4)}
                            for name, value in run.outputs.items()
                        ]
                    )

            self._persist_run(run)
