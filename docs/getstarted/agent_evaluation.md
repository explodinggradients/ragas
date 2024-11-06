# Building and Evaluating a ReAct Agent for Fetching Metal Prices

AI agents are becoming increasingly valuable in domains like finance, e-commerce, and customer support. These agents can autonomously interact with APIs, retrieve real-time data, and perform tasks that align with user goals. Evaluating these agents is crucial to ensure they are effective, accurate, and responsive to different inputs.

In this tutorial, we'll:

1. Build a [ReAct agent](https://arxiv.org/abs/2210.03629) to fetch metal prices.
2. Set up an evaluation pipeline to track key performance metrics.
3. Run and assess the agent's effectiveness with different queries.

Click the [link](https://colab.research.google.com/drive/1WXsVffx6D84ICX-qjjiB6g9AO4XPT_F5?usp=sharing) to open the notebook in Google Colab.

## Prerequisites
- Python 3.8+
- OpenAI API key
- Basic understanding of LangGraph, LangChain and LLMs

## Installing Ragas and Other Dependencies
Install Ragas and Langgraph with pip:


```python
%pip install langgraph==0.2.44
%pip install ragas==0.2.3
%pip install nltk
```

## Building the ReAct Agent

### Initializing the External Components: Setting Up the Metals.Dev API
In this section, we'll walk through how to set up the [metals.dev](https://metals.dev/) API, which provides real-time prices for both precious and industrial metals. The API gives us a simple way to fetch the latest prices for metals like gold, silver, copper, and more.

We selected metals.dev due to its free tier, which provides up to 250 requests per month — a sufficient allowance for this educational tutorials. Below are the steps to get started with the service.

Before you can use the [metals.dev](https://metals.dev/) API, you will need to create an account and get an API key. Here’s how to do that:

1. Go to the metals.dev website.
2. Sign up for a free account by clicking on the Sign Up button.
3. Once you've signed up and logged in, you will be taken to your dashboard where you can find your API key.


```python
import requests
from requests.structures import CaseInsensitiveDict

metals_api_key = "YOUR API KEY"
usage_url = f"https://api.metals.dev/usage?api_key={metals_api_key}"

headers = CaseInsensitiveDict()
headers["Accept"] = "application/json"

api_usage_response = requests.get(usage_url, headers=headers)
print(api_usage_response)
```

    <Response [200]>



```python
api_usage_response.json()
```




    {'status': 'success',
     'timestamp': '2024-11-05T08:42:27.906Z',
     'plan': 'Free',
     'total': 250,
     'used': 5,
     'remaining': 245}




```python
currency = "USD"
unit = "g"
price_url = f"https://api.metals.dev/v1/latest?api_key={metals_api_key}&currency={currency}&unit={unit}"

live_metal_price_response = requests.get(price_url, headers=headers)
print(live_metal_price_response)
```

    <Response [200]>



```python
live_metal_price_response.json()["metals"]
```




    {'gold': 88.0842,
     'silver': 1.0484,
     'platinum': 31.9029,
     'palladium': 34.9698,
     'lbma_gold_am': 88.1364,
     'lbma_gold_pm': 88.1766,
     'lbma_silver': 1.0558,
     'lbma_platinum_am': 32.1186,
     'lbma_platinum_pm': 32.0543,
     'lbma_palladium_am': 35.623,
     'lbma_palladium_pm': 34.8514,
     'mcx_gold': 93.1534,
     'mcx_gold_am': 94.1294,
     'mcx_gold_pm': 94.1294,
     'mcx_silver': 1.1198,
     'mcx_silver_am': 1.1478,
     'mcx_silver_pm': 1.1478,
     'ibja_gold': 93.2605,
     'copper': 0.0099,
     'aluminum': 0.0026,
     'lead': 0.002,
     'nickel': 0.0161,
     'zinc': 0.0031,
     'lme_copper': 0.0096,
     'lme_aluminum': 0.0026,
     'lme_lead': 0.002,
     'lme_nickel': 0.0159,
     'lme_zinc': 0.003}



### Alternative for Quick Access: Using a Predefined JSON Object
If you would like to quickly get started without creating an account, you can bypass the setup process and use the predefined JSON object given below that simulates the API response.


```python
metal_price = {
    "gold": 88.1553,
    "silver": 1.0523,
    "platinum": 32.169,
    "palladium": 35.8252,
    "lbma_gold_am": 88.3294,
    "lbma_gold_pm": 88.2313,
    "lbma_silver": 1.0545,
    "lbma_platinum_am": 31.99,
    "lbma_platinum_pm": 32.2793,
    "lbma_palladium_am": 36.0088,
    "lbma_palladium_pm": 36.2017,
    "mcx_gold": 93.2689,
    "mcx_gold_am": 94.281,
    "mcx_gold_pm": 94.1764,
    "mcx_silver": 1.125,
    "mcx_silver_am": 1.1501,
    "mcx_silver_pm": 1.1483,
    "ibja_gold": 93.2713,
    "copper": 0.0098,
    "aluminum": 0.0026,
    "lead": 0.0021,
    "nickel": 0.0159,
    "zinc": 0.0031,
    "lme_copper": 0.0096,
    "lme_aluminum": 0.0026,
    "lme_lead": 0.002,
    "lme_nickel": 0.0158,
    "lme_zinc": 0.0031,
}
```

### Define the get_metal_price Tool

The get_metal_price tool will be used by the agent to fetch the price of a specified metal. We'll create this tool using the @tool decorator from LangChain.

If you want to use real-time data from the metals.dev API, you can modify the function to make a live request to the API. You will have to uncomment the parts related to the real-time API call.


```python
from langchain_core.tools import tool


# Define the tools for the agent to use
@tool
def get_metal_price(metal_name: str) -> float:
    """Fetches the current per gram price of the specified metal.

    Args:
        metal_name : The name of the metal (e.g., 'gold', 'silver', 'platinum').

    Returns:
        float: The current price of the metal in dollars per gram.

    Raises:
        KeyError: If the specified metal is not found in the data source.
    """
    # # Fetch the latest metal prices from the API
    # metal_price = requests.get(price_url, headers=headers).json()["metals"]
    try:
        metal_name = metal_name.lower().strip()
        if metal_name not in metal_price:
            raise KeyError(
                f"Metal '{metal_name}' not found. Available metals: {', '.join(metal_price['metals'].keys())}"
            )
        return metal_price[metal_name]
    except Exception as e:
        raise Exception(f"Error fetching metal price: {str(e)}")
```

### Binding the Tool to the LLM
With the get_metal_price tool defined, the next step is to bind it to the ChatOpenAI model. This enables the agent to invoke the tool during its execution based on the user's requests allowing it to interact with external data and perform actions beyond its native capabilities.


```python
from langchain_openai import ChatOpenAI

tools = [get_metal_price]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
```

In LangGraph, state plays a crucial role in tracking and updating information as the graph executes. As different parts of the graph run, the state evolves to reflect the changes and contains information that is passed between nodes.

For example, in a conversational system like this one, the state is used to track the exchanged messages. Each time a new message is generated, it is added to the state and the updated state is passed through the nodes, ensuring the conversation progresses logically.

### Defining the State
To implement this in LangGraph, we define a state class that maintains a list of messages. Whenever a new message is produced it gets appended to this list, ensuring that the conversation history is continuously updated.


```python
from langgraph.graph import END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### Defining the should_continue Function
The `should_continue` function determines whether the conversation should proceed with further tool interactions or end. Specifically, it checks if the last message contains any tool calls (e.g., a request for metal prices).

- If the last message includes tool calls, indicating that the agent has invoked an external tool, the conversation continues and moves to the "tools" node.
- If there are no tool calls, the conversation ends, represented by the END state.


```python
# Define the function that determines whether to continue or not
def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

### Calling the Model
The `call_model` function interacts with the Language Model (LLM) to generate a response based on the current state of the conversation. It takes the updated state as input, processes it and returns a model-generated response.


```python
# Define the function that calls the model
def call_model(state: GraphState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

### Creating the Assistant Node
The `assistant` node is a key component responsible for processing the current state of the conversation and using the Language Model (LLM) to generate a relevant response. It evaluates the state, determines the appropriate course of action, and invokes the LLM to produce a response that aligns with the ongoing dialogue.


```python
# Node
def assistant(state: GraphState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### Creating the Tool Node
The `tool_node` is responsible for managing interactions with external tools, such as fetching metal prices or performing other actions beyond the LLM's native capabilities. The tools themselves are defined earlier in the code, and the tool_node invokes these tools based on the current state and the needs of the conversation.


```python
from langgraph.prebuilt import ToolNode

# Node
tools = [get_metal_price]
tool_node = ToolNode(tools)
```

### Building the Graph
The graph structure is the backbone of the agentic workflow, consisting of interconnected nodes and edges. To construct this graph, we use the StateGraph builder which allows us to define and connect various nodes. Each node represents a step in the process (e.g., the assistant node, tool node) and the edges dictate the flow of execution between these steps.


```python
from langgraph.graph import START, StateGraph
from IPython.display import Image, display

# Define a new graph for the agent
builder = StateGraph(GraphState)

# Define the two nodes we will cycle between
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

# Set the entrypoint as `agent`
builder.add_edge(START, "assistant")

# Making a conditional edge
# should_continue will determine which node is called next.
builder.add_conditional_edges("assistant", should_continue, ["tools", END])

# Making a normal edge from `tools` to `agent`.
# The `agent` node will be called after the `tool`.
builder.add_edge("tools", "assistant")

# Compile and display the graph for a visual overview
react_graph = builder.compile()
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```


    
![jpeg](agent_evaluation.jpg)
    


To test our setup, we will run the agent with a query. The agent will fetch the price of copper using the metals.dev API.


```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="What is the price of copper?")]
result = react_graph.invoke({"messages": messages})
```


```python
result["messages"]
```




    [HumanMessage(content='What is the price of copper?', additional_kwargs={}, response_metadata={}, id='84ff2bd1-ac7a-40a2-8252-364619edda3c'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_w3N87nWqw4PvyLWxSsDdBVB1', 'function': {'arguments': '{"metal_name":"copper"}', 'name': 'get_metal_price'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 116, 'total_tokens': 134, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-22a26284-d2b6-4fac-947e-63b39c187f79-0', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'copper'}, 'id': 'call_w3N87nWqw4PvyLWxSsDdBVB1', 'type': 'tool_call'}], usage_metadata={'input_tokens': 116, 'output_tokens': 18, 'total_tokens': 134, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}),
     ToolMessage(content='0.0098', name='get_metal_price', id='01e1b413-f967-460d-b59e-185ab20156ca', tool_call_id='call_w3N87nWqw4PvyLWxSsDdBVB1'),
     AIMessage(content='The price of copper is $0.0098 per gram.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 148, 'total_tokens': 162, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'stop', 'logprobs': None}, id='run-0c987dc9-f02c-4a71-ba45-98f632421311-0', usage_metadata={'input_tokens': 148, 'output_tokens': 14, 'total_tokens': 162, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]



### Converting Messages to Ragas Evaluation Format

In the current implementation, the GraphState stores messages exchanged between the human user, the AI (LLM's responses), and any external tools (APIs or services the AI uses) in a list. Each message is an object in LangChain's format

```python
# Implementation of Graph State
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

Each time a message is exchanged during agent execution, it gets added to the messages list in the GraphState. However, Ragas requires a specific message format for evaluating interactions.

Ragas uses its own format to evaluate agent interactions. So, if you're using LangGraph, you will need to convert the LangChain message objects into Ragas message objects. This allows you to evaluate your AI agents with Ragas’ built-in evaluation tools.

**Goal:**  Convert the list of LangChain messages (e.g., HumanMessage, AIMessage, and ToolMessage) into the format expected by Ragas, so the evaluation framework can understand and process them properly.


```python
from typing import List, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import ragas.messages as r
import json


def convert_to_ragas_trace(
    messages: List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]]
) -> List[Union[r.HumanMessage, r.AIMessage, r.ToolMessage]]:
    """
    Converts a list of LangChain messages into Ragas messages for agent evaluation.

    This function takes a list of LangChain message objects (which may include human messages, AI messages,
    and tool messages) and converts them into Ragas message objects. It captures AI tool calls within each AI message,
    if present, for more comprehensive evaluation using Ragas.

    Args:
        messages (List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]]): A list of LangChain messages
            representing the interaction trace of an AI agent. This may include human messages, system messages, AI messages,
            and tool messages.

    Returns:
        List[Union[r.HumanMessage, r.AIMessage, r.ToolMessage]]: A list of Ragas message objects converted from the
            input LangChain messages, ready for agent evaluation in Ragas.
    """
    ragas_trace = []
    for message in messages:
        if isinstance(message, HumanMessage):
            ragas_trace.append(r.HumanMessage(content=message.content))
        elif isinstance(message, AIMessage):
            tool_calls = []
            if message.additional_kwargs and "tool_calls" in message.additional_kwargs:
                for tool_call in message.additional_kwargs["tool_calls"]:
                    tool_calls.append(
                        r.ToolCall(
                            name=tool_call["function"]["name"],
                            args=json.loads(tool_call["function"]["arguments"]),
                        )
                    )

            if not tool_calls:
                ragas_trace.append(r.AIMessage(content=message.content))
            else:
                ragas_trace.append(
                    r.AIMessage(content=message.content, tool_calls=tool_calls)
                )
        elif isinstance(message, ToolMessage):
            ragas_trace.append(r.ToolMessage(content=message.content))

    return ragas_trace
```


```python
convert_to_ragas_trace(messages=result["messages"])
```




    [HumanMessage(content='What is the price of copper?', metadata=None, type='human'),
     AIMessage(content='', metadata=None, type='ai', tool_calls=[ToolCall(name='get_metal_price', args={'metal_name': 'copper'})]),
     ToolMessage(content='0.0098', metadata=None, type='tool'),
     AIMessage(content='The price of copper is $0.0098 per gram.', metadata=None, type='ai', tool_calls=None)]



## Evaluating the Agent's Performance

For this tutorial, let us evaluate the Agent with the following metrics:

- [Tool call Accuracy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#tool-call-accuracy):ToolCallAccuracy is a metric that can be used to evaluate the performance of the LLM in identifying and calling the required tools to complete a given task.  

- [Agent Goal accuracy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#agent-goal-accuracy): Agent goal accuracy is a metric that can be used to evaluate the performance of the LLM in identifying and achieving the goals of the user. This is a binary metric, with 1 indicating that the AI has achieved the goal and 0 indicating that the AI has not achieved the goal.


First, let us actually run our Agent with a couple of queries, and make sure we have the ground truth labels for these queries.

### Tool Call Accuracy


```python
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample

ragas_trace_sample = convert_to_ragas_trace(messages=result["messages"])

sample = MultiTurnSample(
    user_input=ragas_trace_sample,
    reference_tool_calls=[
        r.ToolCall(name="get_metal_price", args={"metal_name": "copper"})
    ],
)

tool_accuracy_scorer = ToolCallAccuracy()
tool_accuracy_scorer.llm = ChatOpenAI(model="gpt-4o-mini")
await tool_accuracy_scorer.multi_turn_ascore(sample)
```




    1.0



Tool Call Accuracy: 1, because the LLM correctly identified and used the necessary tool (get_metal_price) with the correct parameters (i.e., metal name as "copper").

### Agent Goal Accuracy


```python
messages = [HumanMessage(content="What is the price of 10 grams of gold?")]

result = react_graph.invoke({"messages": messages})
```


```python
result["messages"]
```




    [HumanMessage(content='What is the price of 10 grams of gold?', additional_kwargs={}, response_metadata={}, id='a2cf8c67-c609-4708-a79a-7daf58faff83'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AksI9lzo8sr4Thd1nshXwmef', 'function': {'arguments': '{"metal_name":"gold"}', 'name': 'get_metal_price'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 120, 'total_tokens': 137, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-111c3af8-1205-4e79-b699-457d165f2fb6-0', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'gold'}, 'id': 'call_AksI9lzo8sr4Thd1nshXwmef', 'type': 'tool_call'}], usage_metadata={'input_tokens': 120, 'output_tokens': 17, 'total_tokens': 137, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}),
     ToolMessage(content='88.1553', name='get_metal_price', id='db6dda17-735e-4f58-ac26-61925ba73aa5', tool_call_id='call_AksI9lzo8sr4Thd1nshXwmef'),
     AIMessage(content='The current price of gold is approximately $88.16 per gram. Therefore, the price of 10 grams of gold would be about $881.53.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 151, 'total_tokens': 184, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'stop', 'logprobs': None}, id='run-542c5825-ea52-4b59-842c-07ce7a6636d7-0', usage_metadata={'input_tokens': 151, 'output_tokens': 33, 'total_tokens': 184, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]




```python
ragas_trace_sample = convert_to_ragas_trace(messages=result["messages"])
ragas_trace_sample
```




    [HumanMessage(content='What is the price of 10 grams of gold?', metadata=None, type='human'),
     AIMessage(content='', metadata=None, type='ai', tool_calls=[ToolCall(name='get_metal_price', args={'metal_name': 'gold'})]),
     ToolMessage(content='88.1553', metadata=None, type='tool'),
     AIMessage(content='The current price of gold is approximately $88.16 per gram. Therefore, the price of 10 grams of gold would be about $881.53.', metadata=None, type='ai', tool_calls=None)]




```python
from ragas.dataset_schema import MultiTurnSample
from ragas.metrics import AgentGoalAccuracyWithReference
from ragas.llms import LangchainLLMWrapper


sample = MultiTurnSample(
    user_input=ragas_trace_sample,
    reference="Price of 10 grams of gold",
)

scorer = AgentGoalAccuracyWithReference()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
scorer.llm = evaluator_llm
await scorer.multi_turn_ascore(sample)
```




    1.0



Agent Goal Accuracy: 1, because the LLM correctly achieved the user’s goal of retrieving the price of 10 grams of gold.

## What’s next
🎉 Congratulations! We have learned how to evaluate an agent using the Ragas evaluation framework.
