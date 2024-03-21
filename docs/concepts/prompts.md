# Prompt Objects

Prompts play a crucial role in any language model-based framework and warrant more consideration than mere strings. A well-crafted prompt should include a clear task instruction, articulated in straightforward natural language, comprehensible to any language model. The objective is to compose prompts that are generalizable and do not overly specialize to a specific state of the language model. It's widely recognized that language models exhibit higher accuracy in few-shot scenarios as opposed to zero-shot contexts. To capitalize on this advantage, it is advisable to accompany each prompt with relevant examples.

Prompts in ragas are defined using the `Prompt` class. Each prompt defined using this class will contain.

- `name`: a name given to the prompt. Used to save and identify the prompt.
- `instruction`: The natural language description of the task to be carried out by the LLM
- `examples`: List one or more demonstrations of the task at hand. Using demonstrations converts the task from zero-shot to few-shot which can improve accuracy in most cases.
- `input_keys`:  List of one or more variable names that are used to identify the inputs provided to the LLM.
- `output_key`: Variable name that is used to identify the output
- `output_type`: Output type of the prompt. Can be JSON or string.
- `language`: the language in which demonstrations are written. The default is English.

Let’s create a simple prompt using `Prompt`
```{code-block} python
from ragas.llms.prompt import Prompt

qa_prompt = Prompt(
    name="question_generation",
    instruction="Generate a question for the given answer",
    examples=[
        {
            "answer": "The last Olympics was held in Tokyo, Japan.",
            "context": "The last Olympics was held in Tokyo, Japan. It is held every 4 years",
            "output": {"question":"Where was the last Olympics held?"},
        },
        {
            "answer": "It can change its skin color based on the temperature of its environment.",
            "context": "A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.",
            "output": {"question":"What unique ability does the newly discovered species of frog have?"},
        }
    ],
    input_keys=["answer", "context"],
    output_key="output",
    output_type="json",
)
```

This will create a Prompt class object with the given instruction, examples, and keys. The `output_type` is given as JSON here which will process the example values as JSON strings. This object when created will undergo validations to check if the prompt class criteria are met.

- `instruction` is mandatory and cannot be an empty string.
- `input_keys` and `output_key` are mandatory fields. Multiple `input_keys` can be used but a single `output_key` is accepted.
- `examples` are optional but if provided should contain the input_key and output_keys in the example keys. The example values should match the output_type (dict or json or str).

Prompt objects have the following methods that can be used when evaluating or formatting a prompt object.

- `to_string(self)`

    This method will generate a prompt string from the given object. This string can be directly used as a formatted string with the metrics in the evaluation task.

    ```{code-block} python
    print(qa_prompt.to_string())
    ```

    ```
    Generate a question for the given answer

    answer: "The last Olympics was held in Tokyo, Japan."
    context: "The last Olympics was held in Tokyo, Japan. It is held every 4 years"
    output: {{"question": "Where was the last Olympics held?"}}

    answer: "It can change its skin color based on the temperature of its environment."
    context: "A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment."
    output: {{"question": "What unique ability does the newly discovered species of frog have?"}}

    answer: {answer}
    context: {context}
    output:
    ```

- `format(self, **kwargs)`

    This method will use the parameters passed as keyword arguments to format the prompt object and return a Langchain `PromptValue` object that can be directly used in the evaluation tasks.

    ```{code-block} python
    qa_prompt.format(answer="This is an answer", context="This is a context")
    ```

    ```{code-block} python
    PromptValue(prompt_str='Generate a question for the given answer\n\nanswer: "The last Olympics was held in Tokyo, Japan."\ncontext: "The last Olympics was held in Tokyo, Japan. It is held every 4 years"\noutput: {"question": "Where was the last Olympics held?"}\n\nanswer: "It can change its skin color based on the temperature of its environment."\ncontext: "A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment."\noutput: {"question": "What unique ability does the newly discovered species of frog have?"}\n\nanswer: This is an answer\ncontext: This is a context\noutput: \n')

    ```


- `save(self, cache_dir)`

     This method will save the prompt to the given cache_dir (default `~/.cache`) directory using the value in the `name` variable.

    ```{code-block} python
    qa_prompt.save()
    ```

    The prompts are saved in JSON format to `~/.cache/ragas` by default. One can change this by setting the `RAGAS_CACHE_HOME` environment variable to the desired path. In this example,  the prompt will be saved in `~/.cache/ragas/english/question_generation.json`

- `_load(self, language, name, cache_dir)`

     This method will load the appropriate prompt from the saved directory.

    ```{code-block} python
    from ragas.utils import RAGAS_CACHE_HOME
    Prompt._load(name="question_generation",language="english",cache_dir=RAGAS_CACHE_HOME)
    ```

    ```{code-block} python
    Prompt(name='question_generation', instruction='Generate a question for the given answer', examples=[{'answer': 'The last Olympics was held in Tokyo, Japan.', 'context': 'The last Olympics was held in Tokyo, Japan. It is held every 4 years', 'output': {'question': 'Where was the last Olympics held?'}}, {'answer': 'It can change its skin color based on the temperature of its environment.', 'context': 'A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.', 'output': {'question': 'What unique ability does the newly discovered species of frog have?'}}], input_keys=['answer', 'context'], output_key='output', output_type='JSON')
    ```

    The prompt was loaded from `.cache/ragas/english/question_generation.json`