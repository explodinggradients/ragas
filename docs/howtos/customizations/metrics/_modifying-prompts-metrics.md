# Modifying prompts in metrics

Every metric in Ragas that uses an LLM also uses one or more prompts to generate intermediate results that are used to formulate scores. Prompts can be treated like hyperparameters when using LLM-based metrics. An optimized prompt that suits your domain and use-case can increase the accuracy of your LLM-based metrics by 10-20%. Since optimal prompts depend on the LLM being used, you may want to tune the prompts that power each metric.

Each prompt in Ragas is written using the [BasePrompt][ragas.prompt.metrics.base_prompt.BasePrompt] or [PydanticPrompt][ragas.prompt.pydantic_prompt.PydanticPrompt] classes. Make sure you have an understanding of the [Prompt Object documentation](../../../concepts/components/prompt.md) before going further.

### Understand the prompts of your Metric

For metrics that support prompt customization, Ragas provides access to the underlying prompt objects through the metric instance. Let's look at how to access and modify prompts in the `Faithfulness` metric:

```python
from ragas.metrics.collections import Faithfulness
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Setup dependencies
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric instance
scorer = Faithfulness(llm=llm)

# Access the internal prompt (implementation specific to each metric)
# Most modern metrics have prompts initialized in __init__
print(scorer.prompt)
```
Output
```
<ragas.metrics.collections.faithfulness.util.FaithfulnessPrompt at 0x7f8c41410970>
```



### Generating and viewing the prompt string

Let's view the prompt that will be sent to the LLM:

```python
# For metrics with BasePrompt, you can generate the prompt string
# Here's an example with Faithfulness metric
from ragas.metrics.collections.faithfulness.util import FaithfulnessInput

# Create sample input
sample_input = FaithfulnessInput(
    response="The Eiffel Tower is located in Paris.",
    context="The Eiffel Tower is an iconic iron lattice tower located in Paris, France."
)

# Generate the prompt string
prompt_string = scorer.prompt.to_string(sample_input)
print(prompt_string)
```
Output
```
Your task is to judge the faithfulness of a series of statements based on a given context.
For each statement you must return verdict as 1 if the statement can be directly inferred
based on the context or 0 if the statement can not be directly inferred based on the context.

[Example statements and context shown here...]
```

### Modifying prompts in modern metrics

Modern metrics in Ragas use modular BasePrompt classes that are part of each metric's `util.py` module. To customize a prompt:

1. **Access the prompt**: The prompt is available as an attribute (usually `self.prompt`) on metric instances
2. **Modify the prompt class**: Extend or subclass the prompt to customize instruction or examples
3. **Update the metric**: Pass your custom prompt during metric initialization or modify it afterward

Here's an example with the `FactualCorrectness` metric:

```python
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionInput,
    ClaimDecompositionOutput,
    ClaimDecompositionPrompt,
)

# Create a custom prompt by subclassing
class CustomClaimDecompositionPrompt(ClaimDecompositionPrompt):
    @property
    def instruction(self):
        return """You are an expert at breaking down complex statements into atomic claims.
Break down the input text into clear, verifiable claims.
Only output valid JSON with a "claims" array."""

# Create metric instance and replace prompt
scorer = FactualCorrectness(llm=llm)
scorer.prompt = CustomClaimDecompositionPrompt()

# Now the metric will use the custom prompt
result = await scorer.ascore(
    response="The Eiffel Tower is in Paris and was built in 1889.",
    reference="The Eiffel Tower is located in Paris. It was completed in 1889."
)
```

### Modifying examples in default prompt

Few-shot examples can greatly influence LLM outputs. The examples in default prompts may not reflect your specific domain or use-case. Here's how to modify them:

```python
from ragas.metrics.collections.faithfulness.util import (
    FaithfulnessInput,
    FaithfulnessOutput,
    FaithfulnessPrompt,
    StatementFaithfulnessAnswer,
)

# Create custom prompt with domain-specific examples
class DomainSpecificFaithfulnessPrompt(FaithfulnessPrompt):
    examples = [
        (
            FaithfulnessInput(
                response="Machine learning is a subset of AI that uses statistical techniques.",
                context="Machine learning is a field within artificial intelligence that enables systems to learn from data without being explicitly programmed.",
            ),
            FaithfulnessOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Machine learning is a subset of AI.",
                        reason="This statement is supported by the context which mentions ML as a field within AI.",
                        verdict=1
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Machine learning uses statistical techniques.",
                        reason="While related, the context doesn't explicitly mention statistical techniques.",
                        verdict=0
                    ),
                ]
            ),
        ),
        # Add more examples for your specific domain
    ]

# Update the metric with custom prompt
scorer = Faithfulness(llm=llm)
scorer.prompt = DomainSpecificFaithfulnessPrompt()

# Now evaluate with domain-specific prompts
result = await scorer.ascore(
    response="Neural networks are inspired by biological neurons.",
    context="Artificial neural networks are computing systems loosely inspired by biological neural networks found in animal brains."
)
```

This approach ensures that the LLM has examples that better reflect your domain and evaluation criteria.

### Full prompt customization example

Here's a complete example showing how to verify your customizations:

```python
# Create sample input to test the prompt
sample_input = FaithfulnessInput(
    response="The capital of France is Paris.",
    context="Paris is the capital and most populous city of France."
)

# Generate and view the full prompt string
full_prompt = scorer.prompt.to_string(sample_input)
print("Full Prompt:")
print(full_prompt)
print("\n" + "="*80 + "\n")

# Now use it for evaluation
result = await scorer.ascore(
    response="The capital of France is Paris.",
    context="Paris is the capital and most populous city of France."
)
print(f"Faithfulness Score: {result.value}")
```
