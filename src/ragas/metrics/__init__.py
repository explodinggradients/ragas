import inspect
import sys

from ragas.metrics._answer_correctness import AnswerCorrectness, answer_correctness
from ragas.metrics._answer_relevance import AnswerRelevancy, answer_relevancy
from ragas.metrics._answer_similarity import AnswerSimilarity, answer_similarity
from ragas.metrics._aspect_critic import AspectCritic
from ragas.metrics._context_entities_recall import (
    ContextEntityRecall,
    context_entity_recall,
)
from ragas.metrics._context_precision import (
    ContextPrecision,
    ContextUtilization,
    context_precision,
    context_utilization,
)
from ragas.metrics._context_recall import ContextRecall, context_recall
from ragas.metrics._domain_specific_rubrics import (
    RubricsScoreWithoutReference,
    RubricsScoreWithReference,
    rubrics_score_with_reference,
    rubrics_score_without_reference,
)
from ragas.metrics._faithfulness import Faithfulness, FaithulnesswithHHEM, faithfulness
from ragas.metrics._noise_sensitivity import (
    NoiseSensitivity,
    noise_sensitivity_irrelevant,
    noise_sensitivity_relevant,
)
from ragas.metrics._summarization import SummarizationScore, summarization_score

__all__ = [
    "AnswerCorrectness",
    "answer_correctness",
    "Faithfulness",
    "faithfulness",
    "FaithulnesswithHHEM",
    "AnswerSimilarity",
    "answer_similarity",
    "ContextPrecision",
    "context_precision",
    "ContextUtilization",
    "context_utilization",
    "ContextRecall",
    "context_recall",
    "AspectCritic",
    "AnswerRelevancy",
    "answer_relevancy",
    "ContextEntityRecall",
    "context_entity_recall",
    "SummarizationScore",
    "summarization_score",
    "NoiseSensitivity",
    "noise_sensitivity_irrelevant",
    "noise_sensitivity_relevant",
    "rubrics_score_with_reference",
    "rubrics_score_without_reference",
    "RubricsScoreWithoutReference",
    "RubricsScoreWithReference",
]

current_module = sys.modules[__name__]
ALL_METRICS = [
    obj
    for name, obj in inspect.getmembers(current_module)
    if name in __all__ and not inspect.isclass(obj) and not inspect.isbuiltin(obj)
]
