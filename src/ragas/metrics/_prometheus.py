from __future__ import annotations

import typing as t
import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


class ScoreFeedback(BaseModel):
    feedback: str = Field(..., description="The feedback for the response")
    score: int = Field(..., description="The score given to the response")


class ScoreFeedbackAnswers(BaseModel):
    __root__: t.List[ScoreFeedback]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_score_feedback_output_instructions = get_json_format_instructions(ScoreFeedbackAnswers)
_score_feedback_output_parser = RagasoutputParser(pydantic_object=ScoreFeedbackAnswers)


PROMETHEUS_ABSOLUTE_PROMPT = Prompt(
    name="prometheus_score",
    output_format_instruction=_score_feedback_output_instructions,
    instruction="""Given an instruction, a response, a reference answer, and a score rubric, evaluate the response by providing detailed feedback and assigning a score between 1 and 5. Format the output as JSON with 'feedback' and 'score' fields.""",
    examples=[
        {
            "question": "Struggling with a recent break-up, a person opens up about the intense feelings of loneliness and sadness. They ask for advice on how to cope with the heartbreak and move forward in life.",
            "answer": """I'm genuinely sorry to hear about your break-up. This can be an immensely challenging time filled with intense emotions of loneliness and sorrow. It's important to remember that it's normal to experience these feelings; they are a testament to the deep connection you once had. Firstly, don't hesitate to embrace your emotions, let them flow freely. Ignoring them often complicates the healing process. It's natural to be upset, to shed tears, or to feel angry. This is all part of the journey towards healing. Self-care can be a healing balm in these tough times. This could mean indulging in your favourite book, journaling your thoughts, or even just a quiet cup of tea. Small acts of kindness to oneself can significantly improve your emotional well-being. Also, don't isolate yourself. Surround yourself with supportive individuals who can offer comfort and understanding. This might include friends, family, or even a mental health professional. They can provide much-needed emotional support and strategies to help you cope. Lastly, remember there's no set timeline for healing. It's different for everyone and it's okay to take as long as you need. Keep your focus on the future and the potential it holds. Though it may not seem possible now, you will emerge stronger from this experience. It's okay to ask for help, it's okay to grieve, and it's okay to take it one day at a time. You're not alone in this journey, and there are people ready and willing to help. It's okay to heal at your own pace.""",
            "ground_truth": """I can only imagine how difficult this time must be for you. Heartbreak can feel like an overwhelming wave that washes over you, making you feel lost and alone. It's important to remember that it's okay to feel this way; your feelings are valid and it's natural to grieve the loss of a relationship. Firstly, it's crucial to allow yourself to feel the emotions fully. Suppressing or denying them often only prolongs the healing process. It's okay to cry, to feel sad, or to be angry. These are all part of the healing journey. Engaging in self-care activities can also provide a much-needed boost. This could be anything from taking a warm bath, writing in a journal, or simply drinking a cup of your favorite tea. Small acts of self-love can make a big difference in how you feel. Next, try to surround yourself with supportive people who understand your situation and provide comfort. Friends and family can be a great source of strength in times of heartbreak. If you feel comfortable, you might want to consider seeking professional help. Therapists and counselors are trained to provide assistance and tools to navigate through difficult times like these. Lastly, it's important to remember that it's okay to take your time to heal. Everyone has their own pace and there's no rush. Try to focus on the future and the possibilities it holds. While it may not seem like it now, you will come out stronger and more resilient from this experience. Remember, it's okay to ask for help and it's okay to feel the way you feel. You are not alone in this journey and there are people who care about you and want to help. It's okay to take one day at a time. Healing is a process, and it's okay to move through it at your own pace.""",
            "rubrics": {
                "criteria": "Is the model proficient in applying empathy and emotional intelligence to its responses when the user conveys emotions or faces challenging circumstances?",
                "score1_description": "The model neglects to identify or react to the emotional tone of user inputs, giving responses that are unfitting or emotionally insensitive.",
                "score2_description": "The model intermittently acknowledges emotional context but often responds without sufficient empathy or emotional understanding.",
                "score3_description": "The model typically identifies emotional context and attempts to answer with empathy, yet the responses might sometimes miss the point or lack emotional profundity.",
                "score4_description": "The model consistently identifies and reacts suitably to emotional context, providing empathetic responses. Nonetheless, there may still be sporadic oversights or deficiencies in emotional depth.",
                "score5_description": "The model excels in identifying emotional context and persistently offers empathetic, emotionally aware responses that demonstrate a profound comprehension of the user's emotions or situation."
            },
            "analysis": ScoreFeedbackAnswers.parse_obj(
                [
                    {
                        "feedback": """The response provided shows a high level of empathy and emotional intelligence. It effectively addresses the emotional distress expressed by the user. It acknowledges the user's pain and validates their feelings of loneliness and sadness, which is a crucial aspect of providing empathetic advice. The response also suggests practical steps for coping, such as embracing emotions, practicing self-care, and seeking support from friends, family, or professionals. Furthermore, the response reassures the user that healing is a personal process with no fixed timeline, offering comfort and understanding. It emphasizes the user's worth and potential to overcome the situation, which demonstrates a profound comprehension of the user's emotions and situation. By comparing the score rubric with the provided response, it is clear that the model exhibits an excellent ability to apply empathy and emotional intelligence. The response does not have any deficiencies in emotional depth and successfully meets the criteria for a score of 5.""",
                        "score": 5,
                    }
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "ground_truth", "rubrics"],
    output_key="analysis",
    language="english",
)

class PrometheusAbsolute(MetricWithLLM):
    name = "prometheus_absolute"
    evaluation_mode = EvaluationMode.qga  # Uses question, ground truth, answer

    def __init__(
            self,
            rubrics: Optional[Dict] = None,
            llm: Optional[BaseRagasLLM] = None,
            max_retries: int = 1,
    ):
        super().__init__(llm=llm)
        self.rubrics = rubrics
        self.max_retries = max_retries


    async def _ascore(self, row: Dict, callbacks: t.Callbacks, is_async: bool = False) -> float:
        prompt_value = self._create_prompt(row)

        response = await self.llm.generate(prompt_value, callbacks=callbacks)

        parsed_response = await _score_feedback_output_parser.aparse(
            response.generations[0][0].text, prompt_value, self.llm, self.max_retries
        )

        if parsed_response is None:
            return np.nan

        score = parsed_response.dicts()[0]['score']
        return score

    def _create_prompt(self, row: Dict) -> Prompt:
        return PROMETHEUS_ABSOLUTE_PROMPT.format(
            question=row.get('question', ''),
            answer=row.get('answer', ''),
            ground_truth=row.get('ground_truth', ''),
            rubrics=self.rubrics,
        )

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        PROMETHEUS_ABSOLUTE_PROMPT.adapt(language, self.llm, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        PROMETHEUS_ABSOLUTE_PROMPT.save(cache_dir)
