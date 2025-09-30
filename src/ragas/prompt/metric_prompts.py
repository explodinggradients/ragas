"""
Prompts for migrated metrics that use InstructorLLM.

This module contains all prompt templates for metrics that have been migrated
from LangChain to use direct InstructorLLM calls. These prompts use Python's
str.format() method for variable substitution.
"""

# ============================================================================
# FAITHFULNESS METRIC PROMPTS
# ============================================================================

STATEMENT_GENERATOR_PROMPT = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

--------EXAMPLES-----------
Example 1
Input: {{"question": "Who was Albert Einstein and what is he best known for?", "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."}}
Output: {{"statements": ["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.", "Albert Einstein was best known for developing the theory of relativity.", "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "answer": "{answer}"}}
Output: """

NLI_STATEMENT_PROMPT = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.

--------EXAMPLES-----------
Example 1
Input: {{"context": "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.", "statements": ["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", "John is a dedicated student.", "John has a part-time job."]}}
Output: {{"statements": [{{"statement": "John is majoring in Biology.", "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.", "verdict": 0}}, {{"statement": "John is taking a course on Artificial Intelligence.", "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.", "verdict": 0}}, {{"statement": "John is a dedicated student.", "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.", "verdict": 1}}, {{"statement": "John has a part-time job.", "reason": "There is no information given in the context about John having a part-time job.", "verdict": 0}}]}}

Example 2
Input: {{"context": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.", "statements": ["Albert Einstein was a genius."]}}
Output: {{"statements": [{{"statement": "Albert Einstein was a genius.", "reason": "The context and statement are unrelated", "verdict": 0}}]}}
-----------------------------

Now perform the same with the following input
input: {{"context": "{context}", "statements": {statements_json}}}
Output: """

# ============================================================================
# ANSWER CORRECTNESS METRIC PROMPTS
# ============================================================================

CORRECTNESS_CLASSIFIER_PROMPT = """Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.

--------EXAMPLES-----------
Example 1
Input: {{"question": "What powers the sun and what is it made of?", "answer": ["The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "This fusion process occurs in the sun's core.", "The sun is primarily made of hydrogen and helium."], "ground_truth": ["The sun is powered by nuclear fusion.", "Nuclear fusion in the sun involves hydrogen atoms fusing to form helium.", "This process occurs in the sun's core.", "The sun is primarily composed of hydrogen, with helium being the second most abundant element."]}}
Output: {{"TP": [{{"statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "reason": "This statement is directly supported by the ground truth statements about nuclear fusion and hydrogen-helium conversion."}}, {{"statement": "This fusion process occurs in the sun's core.", "reason": "This statement is directly supported by the ground truth mentioning that the process occurs in the sun's core."}}, {{"statement": "The sun is primarily made of hydrogen and helium.", "reason": "This statement aligns with the ground truth about the sun's composition of hydrogen and helium."}}], "FP": [], "FN": [{{"statement": "Nuclear fusion in the sun involves hydrogen atoms fusing to form helium.", "reason": "This specific detail about hydrogen atoms fusing to form helium is mentioned in ground truth but not explicitly stated in the answer."}}, {{"statement": "The sun is primarily composed of hydrogen, with helium being the second most abundant element.", "reason": "The ground truth provides more specific information about helium being the second most abundant element, which is not explicitly mentioned in the answer."}}]}}
-----------------------------

Now perform the same with the following input
input: {{"question": "{question}", "answer": {answer_json}, "ground_truth": {ground_truth_json}}}
Output: """

# ============================================================================
# FACTUAL CORRECTNESS METRIC PROMPTS
# ============================================================================


def generate_claim_decomposition_prompt(
    atomicity: str, coverage: str, response: str, examples_text: str = ""
) -> str:
    """Generate claim decomposition prompt based on atomicity and coverage levels.

    Args:
        atomicity: Level of atomicity (high/low)
        coverage: Level of coverage (high/low)
        response: The response text to decompose
        examples_text: Pre-formatted examples text

    Returns:
        Formatted prompt string
    """
    return f"""Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
Follow the level of atomicity and coverage as shown in the examples.
{examples_text}
Now perform the same with the following input
input: {{"response": "{response}"}}
Output: """


# Note: NLI_STATEMENT_PROMPT is reused from faithfulness prompts above
# since factual correctness uses the same NLI logic
