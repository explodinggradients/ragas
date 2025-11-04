# AG-UI Integration
Ragas can evaluate agents that stream events via the [AG-UI protocol](https://docs.ag-ui.com/). This notebook shows how to build evaluation datasets, configure metrics, and score AG-UI endpoints.


## Prerequisites
- Install optional dependencies with `pip install "ragas[ag-ui]" langchain-openai python-dotenv nest_asyncio`
- Start an AG-UI compatible agent locally (Google ADK, PydanticAI, CrewAI, etc.)
- Create an `.env` file with your evaluator LLM credentials (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc.)
- If you run this notebook, call `nest_asyncio.apply()` (shown below) so you can `await` coroutines in-place.



```python
# !pip install "ragas[ag-ui]" langchain-openai python-dotenv nest_asyncio

```

## Imports and environment setup
Load environment variables and import the classes used throughout the walkthrough.



```python
import asyncio

from dotenv import load_dotenv
import nest_asyncio
from IPython.display import display
from langchain_openai import ChatOpenAI

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample, MultiTurnSample
from ragas.integrations.ag_ui import (
    evaluate_ag_ui_agent,
    convert_to_ragas_messages,
    convert_messages_snapshot,
)
from ragas.messages import HumanMessage, ToolCall
from ragas.metrics import FactualCorrectness, ToolCallF1
from ragas.llms import LangchainLLMWrapper
from ag_ui.core import (
    MessagesSnapshotEvent,
    TextMessageChunkEvent,
    UserMessage,
    AssistantMessage,
)

load_dotenv()
# Patch the existing notebook loop so we can await coroutines safely
nest_asyncio.apply()

```

## Build single-turn evaluation data
Create `SingleTurnSample` entries when you only need to grade the final answer text.



```python
scientist_questions = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input="Who originated the theory of relativity?",
            reference="Albert Einstein originated the theory of relativity.",
        ),
        SingleTurnSample(
            user_input="Who discovered penicillin and when?",
            reference="Alexander Fleming discovered penicillin in 1928.",
        ),
    ]
)

scientist_questions

```




    EvaluationDataset(features=['user_input', 'reference'], len=2)



## Build multi-turn conversations
For tool-usage metrics, extend the dataset with `MultiTurnSample` and expected tool calls.



```python
weather_queries = EvaluationDataset(
    samples=[
        MultiTurnSample(
            user_input=[HumanMessage(content="What's the weather in Paris?")],
            reference_tool_calls=[
                ToolCall(name="weatherTool", args={"location": "Paris"})
            ],
        )
    ]
)

weather_queries

```




    EvaluationDataset(features=['user_input', 'reference_tool_calls'], len=1)



## Configure metrics and the evaluator LLM
Wrap your grading model with the appropriate adapter and instantiate the metrics you plan to use.



```python
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

qa_metrics = [FactualCorrectness(llm=evaluator_llm)]
tool_metrics = [ToolCallF1()]  # rule-based, no LLM required

```

    /var/folders/8k/tf3xr1rd1fl_dz35dfhfp_tc0000gn/T/ipykernel_93918/2135722072.py:1: DeprecationWarning: LangchainLLMWrapper is deprecated and will be removed in a future version. Use llm_factory instead: from openai import OpenAI; from ragas.llms import llm_factory; llm = llm_factory('gpt-4o-mini', client=OpenAI(api_key='...'))
      evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))


## Evaluate a live AG-UI endpoint
Set the endpoint URL exposed by your agent. Toggle the flags when you are ready to run the evaluations.
In Jupyter/IPython you can `await` the helpers directly once `nest_asyncio.apply()` has been called.



```python
AG_UI_ENDPOINT = "http://localhost:8000/agentic_chat"  # Update to match your agent

RUN_FACTUAL_EVAL = False
RUN_TOOL_EVAL = False

```


```python
async def evaluate_factual():
    return await evaluate_ag_ui_agent(
        endpoint_url=AG_UI_ENDPOINT,
        dataset=scientist_questions,
        metrics=qa_metrics,
        evaluator_llm=evaluator_llm,
        metadata=True,
    )

if RUN_FACTUAL_EVAL:
    factual_result = await evaluate_factual()
    factual_df = factual_result.to_pandas()
    display(factual_df)

```


    Calling AG-UI Agent:   0%|          | 0/2 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/2 [00:00<?, ?it/s]



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>factual_correctness(mode=f1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Who originated the theory of relativity?</td>
      <td>[]</td>
      <td>The theory of relativity was originated by Alb...</td>
      <td>Albert Einstein originated the theory of relat...</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Who discovered penicillin and when?</td>
      <td>[]</td>
      <td>Penicillin was discovered by Alexander Fleming...</td>
      <td>Alexander Fleming discovered penicillin in 1928.</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
async def evaluate_tool_usage():
    return await evaluate_ag_ui_agent(
        endpoint_url=AG_UI_ENDPOINT,
        dataset=weather_queries,
        metrics=tool_metrics,
        evaluator_llm=evaluator_llm,
    )

if RUN_TOOL_EVAL:
    tool_result = await evaluate_tool_usage()
    tool_df = tool_result.to_pandas()
    display(tool_df)

```


    Calling AG-UI Agent:   0%|          | 0/1 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/1 [00:00<?, ?it/s]



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference_tool_calls</th>
      <th>tool_call_f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'What's the weather in Paris?', '...</td>
      <td>[{'name': 'weatherTool', 'args': {'location': ...</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


## Convert recorded AG-UI events
Use the conversion helpers when you already have an event log to grade offline.



```python
events = [
    TextMessageChunkEvent(
        message_id="assistant-1",
        role="assistant",
        delta="Hello from AG-UI!",
    )
]

messages_from_stream = convert_to_ragas_messages(events, metadata=True)

snapshot = MessagesSnapshotEvent(
    messages=[
        UserMessage(id="msg-1", content="Hello?"),
        AssistantMessage(id="msg-2", content="Hi! How can I help you today?"),
    ]
)

messages_from_snapshot = convert_messages_snapshot(snapshot)

messages_from_stream, messages_from_snapshot

```




    ([AIMessage(content='Hello from AG-UI!', metadata={'timestamp': None, 'message_id': 'assistant-1'}, type='ai', tool_calls=None)],
     [HumanMessage(content='Hello?', metadata=None, type='human'),
      AIMessage(content='Hi! How can I help you today?', metadata=None, type='ai', tool_calls=None)])




```python

```
