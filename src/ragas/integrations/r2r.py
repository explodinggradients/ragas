import typing as t
import warnings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import Metric


class RagasEvaluator:
    """
    Evaluates R2R RAG using the Ragas framework with configurable metrics, LLMs, and embeddings.
    """

    def __init__(
        self,
        metrics: t.List[Metric],
        evaluator_llm: t.Optional[BaseRagasLLM] = None,
        evaluator_embeddings: t.Optional[BaseRagasEmbeddings] = None,
    ):
        """
        Initializes the RagasEvaluator with specified metrics, LLM, and embeddings.

        Parameters
        ----------
        metrics : List[Metric]
            A list of Ragas metrics to use for evaluation.
        evaluator_llm : Optional[BaseRagasLLM], default=None
            An optional ragas LLM wrapper to use for evaluation.
        evaluator_embeddings : Optional[BaseRagasEmbeddings], default=None
            Optional ragas embeddings model wrapper for similarity-based evaluation.
        """
        self.metrics = metrics
        self.evaluator_llm = evaluator_llm
        self.evaluator_embeddings = evaluator_embeddings

    def _process_search_results(
        self, search_results: t.Dict[str, t.List]
    ) -> t.List[str]:
        """
        Extracts relevant text from search results while issuing warnings for unsupported result types.

        Parameters
        ----------
        search_results : Dict[str, List]
            A dictionary containing search results from different retrieval sources.

        Returns
        -------
        List[str]
            A list of extracted text snippets from chunk and web search results.
        """
        retrieved_contexts = []

        for key in ["graph_search_results", "context_document_results"]:
            if search_results.get(key) and len(search_results[key]) > 0:
                warnings.warn(
                    f"{key} are not included in the aggregated `retrieved_context` for Ragas evaluations."
                )

        for result in search_results.get("chunk_search_results", []):
            text = result.get("text")
            if text:
                retrieved_contexts.append(text)

        for result in search_results.get("web_search_results", []):
            text = result.get("snippet")
            if text:
                retrieved_contexts.append(text)

        return retrieved_contexts

    def _process_messages(
        self, query: str, messages: t.List[t.Dict[str, t.Any]]
    ) -> SingleTurnSample:
        """
        Processes multi-turn messages to create a SingleTurnSample.

        Parameters
        ----------
        query : str
            The user query.
        messages : List[Dict[str, Any]]
            A list of message dictionaries containing dialogue history.

        Returns
        -------
        SingleTurnSample
            A processed sample ready for evaluation.

        Raises
        ------
        NotImplementedError
            This method needs to be implemented.
        """
        raise NotImplementedError()

    def evaluate(
        self,
        query: str,
        response: t.Dict[str, t.Any],
        reference: t.Optional[str] = None,
        reference_contexts: t.Optional[t.List[str]] = None,
        rubrics: t.Optional[t.Dict[str, str]] = None,
    ) -> EvaluationResult:
        """
        Evaluates a single R2R RAG response using the provided metrics and evaluator.

        Parameters
        ----------
        query : str
            The user query.
        response : Dict[str, Any]
            The model-generated response containing either 'completion' or 'message' keys.
        reference : Optional[str], default=None
            The ground truth response for comparison.
        reference_contexts : Optional[List[str]], default=None
            Additional reference contexts for evaluation.
        rubrics : Optional[Dict[str, str]], default=None
            Evaluation rubrics specifying additional constraints.

        Returns
        -------
        EvaluationResult
            The result of the evaluation containing metric scores.

        Raises
        ------
        ValueError
            If the response does not contain the required keys.
        """
        response_content = response.get("results")
        if not response_content:
            raise ValueError("Response must contain a `results` key.")

        if "completion" in response_content:
            single_turn_sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=self._process_search_results(
                    search_results=response_content.get("search_results", {})
                ),
                reference_contexts=reference_contexts,
                response=response_content.get("completion"),
                reference=reference,
                rubrics=rubrics,
            )
        elif "message" in response_content:
            single_turn_sample = self._process_messages(
                query, response_content.get("message")
            )
        else:
            raise ValueError(
                "Response must contain either a `completion` or `message` key."
            )

        result = evaluate(
            dataset=EvaluationDataset([single_turn_sample]),
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        )
        return result

    def evaluate_dataset(
        self,
        queries: t.List[str],
        responses: t.List[t.Dict[str, t.Any]],
        references: t.Optional[t.List[str]] = None,
        reference_contexts: t.Optional[t.List[t.List[str]]] = None,
        rubrics: t.Optional[t.List[t.Dict[str, str]]] = None,
    ) -> EvaluationResult:
        """
        Evaluates a dataset of multiple query-response pairs.

        Parameters
        ----------
        queries : List[str]
            A list of user queries.
        responses : List[Dict[str, Any]]
            A list of model-generated responses.
        references : Optional[List[str]], default=None
            A list of ground truth responses.
        reference_contexts : Optional[List[List[str]]], default=None
            A list of additional reference contexts for each query.
        rubrics : Optional[List[Dict[str, str]]], default=None
            Evaluation rubrics specifying additional constraints.

        Returns
        -------
        EvaluationResult
            The aggregated result of the evaluation.

        Raises
        ------
        ValueError
            If the input lists have mismatched lengths or are empty.
        """
        if not responses:
            raise ValueError("Response list cannot be empty.")

        args = {
            "queries": queries,
            "responses": responses,
            "references": references,
            "reference_contexts": reference_contexts,
            "rubrics": rubrics,
        }
        first_len = len(next(lst for lst in args.values() if lst is not None))

        for name, lst in args.items():
            if lst is not None and len(lst) != first_len:
                raise ValueError(f"Length mismatch: '{name}' has a different length.")

        samples = []
        for idx in range(first_len):
            retrieved_contexts = self._process_search_results(
                search_results=responses[idx]
                .get("results", {})
                .get("search_results", {})
            )
            ragas_response = responses[idx].get("results", {}).get("completion")

            sample = SingleTurnSample(
                user_input=queries[idx],
                retrieved_contexts=retrieved_contexts,
                reference_contexts=(
                    reference_contexts[idx] if reference_contexts else None
                ),
                response=ragas_response,
                reference=references[idx] if references else None,
                rubrics=rubrics[idx] if rubrics else None,
            )
            samples.append(sample)

        return evaluate(
            EvaluationDataset(samples),
            self.metrics,
            self.evaluator_llm,
            self.evaluator_embeddings,
        )
