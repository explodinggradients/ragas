from ragas.llms.prompt import Prompt

common_themes_from_summaries = Prompt(
    name="get_common_themes",
    instruction="Get common themes from the following summaries",
    examples=[
        {
            "summaries":"""""",
            "themes":[],
        }
    ],
    input_keys=["summaries"],
    output_key="themes",
    output_type="json",
    language="english",
)

abstract_question_from_theme = Prompt(
    name="abstract question generation",
    instruction="Generate abstract question from given theme that can be answered document containing given summaries",
    examples=[
        {
            "theme":[""],
            "summaries":"",
            "questions":[]
        }
    ],
    input_keys=["theme", "summaries"],
    output_key="questions",
    output_type="json",
)

question_answering = Prompt(
    name="question_answering",
    instruction="Answer the given question only using information present in the given texts",
    examples=[
        {"question":"",
        "texts":"",
        "answer":""}
    ],
    input_keys=["question", "texts"],
    output_key="answer",
    output_type="str",
    
)


critic_question = Prompt(
    name="critic_question",
    instruction="Critic the synthetically generated question based on following rubrics",
    examples=[
        {
            "question":"",
            "feedback":""
        }
    ],
    input_keys=["question"],
    output_key="feedback",
    output_type="str",
)