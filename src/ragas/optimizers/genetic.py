import typing as t

from langchain_core.callbacks import Callbacks
from pydantic import BaseModel

from ragas.loaders import SampleAnnotation, SingleMetricAnnotation
from ragas.losses import Loss
from ragas.metrics.base import MetricWithLLM
from ragas.optimizers.base import Optimizer
from ragas.prompt import PydanticPrompt


class FormattedExamples(BaseModel):
    examples: t.List[t.Tuple[str, t.Any]]

    @classmethod
    def from_examples(
        cls, examples: t.List[t.Dict[t.Dict[str, t.Any], t.Dict[str, t.Any]]]
    ) -> "FormattedExamples":

        formated_examples = []
        for example in examples:
            input_, output = list(example.items())[0]
            input_ = "".join(f"\n{key}:\n\t{val}\n" for key, val in input_.items())
            formated_examples.append((input_, output))

        return cls(examples=formated_examples)


class OutputInstruction(BaseModel):
    instruction: str


class ReverseEngineerPrompt(PydanticPrompt[FormattedExamples, OutputInstruction]):
    name: str = "reverse_engineer"
    instruction: str = (
        "Given a set of (input containing (user_input, response, reference, etc), expected output) pairs that were manually annotated, guess and generate the instruction given to the annotator."
    )
    input_model = FormattedExamples
    output_model = OutputInstruction


class ParentPrompts(BaseModel):
    parent_1: str
    parent_2: str


class CrossOverPrompt(PydanticPrompt[ParentPrompts, OutputInstruction]):
    name: str = "crossover"
    instruction: str = (
        "You are a mutator who is familiar with the concept of cross-over in genetic algorithm, namely "
        "combining the genetic information of two parents to generate new offspring. Given two parent "
        "prompts, you will perform a cross-over to generate an offspring prompt that covers the same "
        "semantic meaning as both parents."
    )
    input_model = ParentPrompts
    output_model = OutputInstruction
    examples = [
        (
            ParentPrompts(
                parent_1="Now you are a categorizer, your mission is to ascertain the sentiment of the provided text, either favorable or unfavorable.",
                parent_2="Assign a sentiment label to the given sentence from [’negative’, ’positive’] and return only the label without any other text.",
            ),
            OutputInstruction(
                instruction="Your mission is to ascertain the sentiment of the provided text and assign a sentiment label from [’negative’, ’positive’].",
            ),
        )
    ]


class FeedbackExample(BaseModel):
    input: str
    output: t.Dict[str, t.Any]
    expected_output: t.Dict[str, t.Any]


class FeedbackMutationInput(BaseModel):
    instruction: str
    examples: t.List[FeedbackExample]


class FeedbackMutationOutput(BaseModel):
    feedbacks: t.List[str]


class FeedbackMutationPrompt(
    PydanticPrompt[FeedbackMutationInput, FeedbackMutationOutput]
):
    name: str = "feedback_mutation"
    instruction: str = (
        "You're an expert reviewer. Given an instruction and a set of (input  containing (user_input, response, reference), output, expected_output) examples, give maximum 3 feedbacks on how the instruction can be improved to correct the mistakes in incorrect outputs and reach expected output."
        "Do not provide the feedback to add examples with the instruction."
    )
    input_model = FeedbackMutationInput
    output_model = FeedbackMutationOutput


class FeedbackMutationPromptInput(BaseModel):
    instruction: str
    feedbacks: t.List[str]


class FeedbackMutationPromptGeneration(
    PydanticPrompt[FeedbackMutationPromptInput, OutputInstruction]
):
    name: str = "feedback_mutation_generation"
    instruction: str = (
        "You are a mutator. Given an instruction and a set of feedbacks on how the instruction can be improved generate a new instruction that incorporates the feedback."
    )
    input_model = FeedbackMutationPromptInput
    output_model = OutputInstruction


class GeneticOptimizer(Optimizer):
    """
    A genetic algorithm optimizer that balances exploration and exploitation.
    """

    reverse_engineer_prompt = ReverseEngineerPrompt()

    def optimize(
        self,
        dataset: SingleMetricAnnotation,
        loss: Loss,
        config: t.Dict[t.Any, t.Any],
        callbacks: Callbacks,
    ) -> MetricWithLLM:

        if self.metric is None:
            raise ValueError("No metric provided for optimization.")

        if self.llm is None:
            raise ValueError("No llm provided for optimization.")

        # max_steps = config.get("max_steps", 100
        return self.metric

    def _initialize_population(
        self, dataset: SingleMetricAnnotation, population_size: int
    ) -> t.List[str]:

        candidates = []
        dataset = dataset.filter(lambda x: x["is_accepted"])
        batches = dataset.stratified_batches(
            batch_size=population_size,
            stratify_key="metric_output",
            replace=False,
            drop_last_batch=False,
        )
        for batch in batches:
            candidate = self._reverse_engineer_instruction(batch)
            candidates.append(candidate)

    async def _reverse_engineer_instruction(
        self, batch: t.List[SampleAnnotation]
    ) -> t.Dict[str, str]:

        prompt_annotations = {key: [] for key in batch[0]["prompts"].keys()}
        candidates = {}
        for sample in batch:
            input_ouputs = sample["prompts"]
            for name, example in input_ouputs.items():
                input_ = {
                    key: val
                    for key, val in example["prompt_input"].items()
                    if val is not None
                }
                output = (
                    example["edited_output"]
                    if example["edited_output"]
                    else example["prompt_output"]
                )
                prompt_annotations[name].append({"input": input_, "output": output})

        for prompt_name, examples in prompt_annotations.items():
            formatted_examples = FormattedExamples.from_examples(examples)
            instruction = await self.reverse_engineer_prompt.generate(
                data=formatted_examples, llm=self.llm
            )
            candidates[prompt_name] = instruction

        return candidates

    def _cross_over(self, parent_1: str, parent_2: str) -> str:
        return "instruction"
