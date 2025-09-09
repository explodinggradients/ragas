from __future__ import annotations

import logging
import typing as t
import warnings

from ragas.dataset_schema import EvaluationDataset

if t.TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def _process_search_results(search_results: t.Dict[str, t.List]) -> t.List[str]:
    """
    Extracts relevant text from search results while issuing warnings for unsupported result types.

    Parameters
    ----------
    search_results : Dict[str, List]
        A r2r result object of an aggregate search operation.

    Returns
    -------
    List[str]
        A list of extracted text from aggregate search result.
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


def transform_to_ragas_dataset(
    user_inputs: t.Optional[t.List[str]] = None,
    r2r_responses: t.Optional[t.List] = None,
    reference_contexts: t.Optional[t.List[str]] = None,
    references: t.Optional[t.List[str]] = None,
    rubrics: t.Optional[t.List[t.Dict[str, str]]] = None,
) -> EvaluationDataset:
    """
    Converts input data into a Ragas EvaluationDataset, ensuring flexibility
    for cases where only some lists are provided.

    Parameters
    ----------
    user_inputs : Optional[List[str]]
        List of user queries.
    r2r_responses : Optional[List]
        List of responses from the R2R client.
    reference_contexts : Optional[List[str]]
        List of reference contexts.
    references : Optional[List[str]]
        List of reference answers.
    rubrics : Optional[List[Dict[str, str]]]
        List of evaluation rubrics.

    Returns
    -------
    EvaluationDataset
        A dataset containing structured evaluation samples.

    Raises
    ------
    ValueError
        If provided lists (except None ones) do not have the same length.
    """

    # Collect only the non-None lists
    provided_lists = {
        "user_inputs": user_inputs or [],
        "r2r_responses": r2r_responses or [],
        "reference_contexts": reference_contexts or [],
        "references": references or [],
        "rubrics": rubrics or [],
    }

    # Find the maximum length among provided lists
    max_len = max(len(lst) for lst in provided_lists.values())

    # Ensure all provided lists have the same length
    for key, lst in provided_lists.items():
        if lst and len(lst) != max_len:
            raise ValueError(
                f"Inconsistent length for {key}: expected {max_len}, got {len(lst)}"
            )

    # Create samples while handling missing values
    samples = []
    for i in range(max_len):
        sample = {
            "user_input": user_inputs[i] if user_inputs else None,
            "retrieved_contexts": (
                _process_search_results(
                    r2r_responses[i].results.search_results.as_dict()
                )
                if r2r_responses
                else None
            ),
            "reference_contexts": reference_contexts[i] if reference_contexts else None,
            "response": (
                r2r_responses[i].results.generated_answer if r2r_responses else None
            ),
            "reference": references[i] if references else None,
            "rubrics": rubrics[i] if rubrics else None,
        }

        samples.append(sample)

    return EvaluationDataset.from_list(data=samples)
