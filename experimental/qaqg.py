from __future__ import annotations

import re
import typing as t

from llms import llm
from metrics import GenerationMetric

QUESTION_GENERATION = """Given a text, extract {} noun phrases and create questions for each based on given text.
text: Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. Best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
A: Germany
Q: Where was Albert Einstein born?
A: theory of relativity
Q: What is Albert Einstein best known for?
text: {}
"""

QUESTION_ANSWERING = """Given a text and set of questions, answer the questions
text: Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. Best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
questions: Where was Albert Einstein born?\n\nWhat is Albert Einstein best known for?
answers:Germany\n\ntheory of relativity
text: {}
questions:{}
answers:"""

ANSWER_VERIFICATION = """Given a set of questions, correct answer and student's answer return the number of questions incorrectly answered by student.
Where was Albert Einstein born?\nCorrect answer: Germany\nStudent answer:India\n\n
What is Albert Einstein best known for?\nCorrect answer:  theory of relativity\nStudent answer: theory of relativity\n\n
Number of incorrect answers:1
{}
Number of incorrect answers:"""


def QAQG_fun(questions: list[str], contexts: list[list[str]], answers: list[str]):
    """
    returns number of factual inconsistencies.
    """

    def answer_ver(qstn, answer, cand):
        return f"{qstn}\nCorrect answer: {answer}\nStudent answer: {cand}"

    num = len(answer.split(".")) - 1
    prompt = QUESTION_GENERATION.format(num, answer)
    output = llm(prompt)
    qa_pairs = [
        re.sub(r"A:|Q:", "", x).strip()
        for item in output["choices"][0]["text"].strip().split("\n\n")
        for x in item.split("\n")
    ]
    qa_pairs = [tuple(qa_pairs[i : i + 2]) for i in range(0, len(qa_pairs), 2)]
    print(qa_pairs)
    questions = "\n\n".join([qstn for ans, qstn in qa_pairs])
    prompt = QUESTION_ANSWERING.format(context, questions)
    answers = llm(prompt)["choices"][0]["text"].split("\n\n")

    prompt = "\n\n".join(
        [answer_ver(qstn, ans, cand) for (ans, qstn), cand in zip(qa_pairs, answers)]
    )
    output = llm(ANSWER_VERIFICATION.format(prompt))["choices"][0]["text"].strip()
    return int(output)
