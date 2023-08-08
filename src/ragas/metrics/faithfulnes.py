from __future__ import annotations

import typing as t
from dataclasses import dataclass

from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from tqdm import tqdm

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.metrics.llms import generate

if t.TYPE_CHECKING:
    from datasets import Dataset

#################
# NLI Score
#################
LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
    """\
Given a question and answer, create one or more statements from answer.
question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements:\nAlbert Einstein was born in Germany.\nAlbert Einstein was best known for his theory of relativity.
question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alochol
statements:\nCadmium Chloride is slightly soluble in alcohol.
question: Were Shahul and Jithin of the same nationality?
answer: They were from different countries.
statements:\nShahul and Jithin were from different countries.
question:{question}
answer: {answer}
statements:\n"""  # noqa: E501
)


NLI_STATEMENTS_MESSAGE = HumanMessagePromptTemplate.from_template(
    """
Prompt: Natural language inference
Consider the given context and following statements, then determine whether they are supported by the information present in the context.Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

Context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
statements:\n1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.\n
Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
Final verdict for each statement in order: No. No. Yes. No. Yes.
context:\n{context}
statements:\n{statements}
Answer:
"""  # noqa: E501
)


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    batch_size: int = 15

    def init_model(self: t.Self):
        pass

    def score(self: t.Self, dataset: Dataset) -> Dataset:
        assert self.llm is not None, "LLM not initialized"

        scores = []
        with trace_as_chain_group(f"ragas_{self.name}") as score_group:
            for batch in tqdm(self.get_batches(len(dataset))):
                score = self._score_batch(dataset.select(batch), callbacks=score_group)
                scores.extend(score)

        return dataset.add_column(self.name, scores)  # type: ignore

    def _score_batch(
        self: t.Self,
        ds: Dataset,
        callbacks: CallbackManager,
        callback_group_name: str = "batch",
    ) -> list[float]:
        """
        returns the NLI score for each (q, c, a) pair
        """

        question, answer, contexts = ds["question"], ds["answer"], ds["contexts"]
        prompts = []

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for q, a in zip(question, answer):
                human_prompt = LONG_FORM_ANSWER_PROMPT.format(question=q, answer=a)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = generate(prompts, self.llm, callbacks=batch_group)
            list_statements: list[list[str]] = []
            for output in result.generations:
                # use only the first generation for each prompt
                statements = output[0].text.split("\n")
                list_statements.append(statements)

            prompts = []
            for context, statements in zip(contexts, list_statements):
                statements_str: str = "\n".join(
                    [f"{i+1}.{st}" for i, st in enumerate(statements)]
                )
                contexts_str: str = "\n".join(context)
                human_prompt = NLI_STATEMENTS_MESSAGE.format(
                    context=contexts_str, statements=statements_str
                )
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = generate(prompts, self.llm, callbacks=batch_group)
            outputs = result.generations

            scores = []
            final_answer = "Final verdict for each statement in order:"
            final_answer = final_answer.lower()
            for i, output in enumerate(outputs):
                output = output[0].text.lower().strip()
                if output.find(final_answer) != -1:
                    output = output[output.find(final_answer) + len(final_answer) :]
                    score = sum(
                        0 if "yes" in answer else 1
                        for answer in output.strip().split(".")
                        if answer != ""
                    )
                    score = score / len(list_statements[i])
                else:
                    score = max(0, output.count("verdict: no")) / len(
                        list_statements[i]
                    )

                scores.append(1 - score)

        return scores


faithfulness = Faithfulness()
