from __future__ import annotations

import typing as t

from llms import llm, llm_async
from metrics import GenerationMetric

QUESTION_ANSWER_STMNT = """Given a question and answer, create a statement.
question: Who is the president of India?
answer: Narendra Modi
statement: Narendara Modi is the president of India.
question: Which magazine was started first Arthur's Magazine or Women's Magazine?
answer: Arthur's Magazine
statement: Arthur's Magazine started before Women's magazine. 
question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alochol
statement: Cadmium Chloride is slightly soluble in alcohol.
question: Were Shahul and Jithin of the same nationality?
answer: They were from different countries.
statement: Shahul and Jithin were from different countries.
question: {}
answer: {}
statemtent:"""

ANSWER_STMNT = """
Given a question and answer, create one or more statements from answer.
question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements: Albert Einstein was born in Germany.\n\nAlbert Einstein was best known for his theory of relativity.
question:{}
answer: {}
statements:
"""

VERIFY = """
Given a context and set of statements separated by '.', Answer YES for each statement if it is supported by context and NO if not.
context: Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. Best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements: Albert Einstein was born in India. Albert Einstein was best known for his theory of relativity.
answer: NO. YES. 
context: {}
statements: {}
answer:"""
DICT = {"YES": 0, "NO": 1}


class NLIScore(GenerationMetric):
    @property
    def name(self):
        return "NLI_score"

    @property
    def is_batchable(self: t.Self):
        return True

    def init_model(self: t.Self):
        pass

    def score(
        self: t.Self,
        questions: list[str],
        contexts: list[list[str]],
        answers: list[str],
    ):
        """
        returns the NLI score for each (q, c, a) pair
        """

        prompts = []
        for question, answer in zip(questions, answers):
            ## single phrase answer
            if (len(answer.split()) < 4) or (len(answer.split(".")) == 1):
                prompt = QUESTION_ANSWER_STMNT.format(question, answer)
                prompts.append(prompt)
            ## long form
            else:
                prompt = ANSWER_STMNT.format(question, answer)
                prompts.append(prompt)

        response = llm(prompts)
        usage = response["usage"]
        print(usage)
        list_statements = []
        for output in response["choices"]:
            statements = output["text"].split("\n\n")
            list_statements.append(statements)

        # print(list_statements)

        ## verify
        prompts = []
        for context, statements in zip(contexts, list_statements):
            prompt = VERIFY.format(context, ". ".join(statements))
            prompts.append(prompt)

        response = llm(prompts)
        outputs = response["choices"]
        usage = response["usage"]
        print(usage)

        scores = []
        for i, output in enumerate(outputs):
            score = sum(
                [DICT[key.strip()] for key in output["text"].split(".") if key != ""]
            ) / len(list_statements[i])
            scores.append(1 - score)

        return scores


NLI = NLIScore()
