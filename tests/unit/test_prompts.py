from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_relevancy import ContextRelevancy
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics.critique import AspectCritique
from ragas.metrics._faithfulness import Faithfulness

from langchain.prompts import HumanMessagePromptTemplate

test_prompt = 'THIS IS A TEST PROMPT'

answer_relevancy = AnswerRelevancy(question_generation_prompt=test_prompt)
context_precision = ContextPrecision(context_precision_prompt=test_prompt)
context_relevancy = ContextRelevancy(context_relevance_prompt=test_prompt)
context_recall = ContextRecall(context_recall_prompt=test_prompt)
critique = AspectCritique(name="test_critique", definition="This is a test critique", critique_prompt=test_prompt)
faithfulness = Faithfulness(long_answer_prompt=test_prompt, nli_statements_prompt=test_prompt)


def test_prompts():

    assert answer_relevancy.question_generation_prompt==test_prompt, "Answer relevancy custom prompt error"
    assert context_precision.context_precision_prompt==test_prompt, "Context precision custom prompt error"
    assert context_relevancy.context_relevance_prompt==test_prompt, "Context relevancy custom prompt error"
    assert context_recall.context_recall_prompt==test_prompt, "Context recall custom prompt error"
    assert critique.critique_prompt==test_prompt, "Critique custom prompt error"
    assert faithfulness.long_answer_prompt==test_prompt, "Faithfulness custom prompt error"
    assert faithfulness.nli_statements_prompt==test_prompt, "Faithfulness custom prompt error"