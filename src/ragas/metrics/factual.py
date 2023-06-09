from __future__ import annotations

import typing as t
from dataclasses import dataclass

from datasets import concatenate_datasets
from tqdm import tqdm

from ragas.metrics.base import Metric
from ragas.metrics.llms import openai_completion

if t.TYPE_CHECKING:
    from datasets import Dataset

#################
# NLI Score
#################
LONG_FORM_ANSWER = """
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
question:{}
answer: {}
statements:\n"""  # noqa: E501

NLI_STATEMENTS = """
Prompt: Natural language inference
Consider the following context:
Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
Now, read the following statements and determine whether they are supported by the information present in the context. Provide a brief explanation for each statement. Also provide a Final Answer (Yes/No) at the end. 
statements:\n1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.\n
Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology. So answer is No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.So answer is No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.So answer is Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job. So answer is No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming.So answer is Yes.
Final answer: No. No. Yes. No. Yes.
context:\n{}
statements:\n{}
Now, read the following statements and determine whether they are supported by the information present in the context. Provide a brief explanation for each statement. Also provide a Final Answer (Yes/No) at the end. 
Answer:
"""  # noqa: E501


@dataclass
class Factuality(Metric):
    batch_size: int = 15

    @property
    def name(self):
        return "factuality"

    def init_model(self: t.Self):
        pass

    def score(self: t.Self, dataset: Dataset) -> Dataset:
        scores = []
        for batch in tqdm(self.get_batches(len(dataset))):
            score = self._score_batch(dataset.select(batch))
            scores.append(score)

        return concatenate_datasets(scores)

    def _score_batch(self: t.Self, ds: Dataset) -> Dataset:
        """
        returns the NLI score for each (q, c, a) pair
        """
        question, answer, contexts = ds["question"], ds["answer"], ds["contexts"]
        prompts = []
        for q, a in zip(question, answer):
            prompt = LONG_FORM_ANSWER.format(q, a)
            prompts.append(prompt)

        response = openai_completion(prompts)
        list_statements: list[list[str]] = []
        for output in response["choices"]:  # type: ignore
            statements = output["text"].split("\n")
            list_statements.append(statements)

        prompts = []
        for context, statements in zip(contexts, list_statements):
            statements_str: str = "\n".join(
                [f"{i+1}.{st}" for i, st in enumerate(statements)]
            )
            contexts_str: str = "\n".join(context)
            prompt = NLI_STATEMENTS.format(contexts_str, statements_str)
            prompts.append(prompt)

        response = openai_completion(prompts)
        outputs = response["choices"]  # type: ignore

        scores = []
        for i, output in enumerate(outputs):
            output = output["text"].lower().strip()
            if output.find("final answer:") != -1:
                output = output[output.find("final answer:") + len("final answer:") :]
                score = sum(
                    0 if "yes" in answer else 1
                    for answer in output.strip().split(".")
                    if answer != ""
                )
                score = score / len(list_statements[i])
            else:
                score = max(0, output.count("so answer is no")) / len(
                    list_statements[i]
                )

            scores.append(1 - score)

        return ds.add_column(f"{self.name}", scores)  # type: ignore


factuality = Factuality()
