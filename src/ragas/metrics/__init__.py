from ragas.metrics.answer_correctness import AnswerCorrectness, answer_correctness
from ragas.metrics.answer_relevance import AnswerRelevancy, answer_relevancy
from ragas.metrics.answer_similarity import AnswerSimilarity, answer_similarity
from ragas.metrics.context_precision import (
    ContextPrecision,
    ContextRelevancy,
    context_precision,
)
from ragas.metrics.context_recall import ContextRecall, context_recall
from ragas.metrics.critique import AspectCritique
from ragas.metrics.faithfulness import Faithfulness, faithfulness

DEFAULT_METRICS = [answer_relevancy, context_precision, faithfulness, context_recall]

# TODO: remove context_relevancy, ContextRelevancy after 0.1.0
__all__ = [
    "Faithfulness",
    "faithfulness",
    "AnswerRelevancy",
    "answer_relevancy",
    "AnswerSimilarity",
    "answer_similarity",
    "AnswerCorrectness",
    "answer_correctness",
    "ContextRelevancy",
    "context_precision",
    "ContextPrecision",
    "context_precision",
    "AspectCritique",
    "ContextRecall",
    "context_recall",
]
