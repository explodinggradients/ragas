# Building and Evaluating a ReAct Agent for Fetching Metal Prices

AI agents are becoming increasingly valuable in domains like finance, e-commerce, and customer support. These agents can autonomously interact with APIs, retrieve real-time data, and perform tasks that align with user goals. Evaluating these agents is crucial to ensure they are effective, accurate, and responsive to different inputs.

In this tutorial, we'll:

1. Build a [ReAct agent](https://arxiv.org/abs/2210.03629) to fetch metal prices.
2. Set up an evaluation pipeline to track key performance metrics.
3. Run and assess the agent's effectiveness with different queries.

Click the [link](https://colab.research.google.com/drive/1IsmUfCVD_pVQ7HGsKj97BXOZBGXTIF8m?usp=sharing) to open the notebook in Google Colab.

## Prerequisites
- Python 3.8+
- OpenAI API key
- Basic understanding of LangGraph, LangChain and LLMs

## Installing Ragas and Other Dependencies
Install Ragas and Langgraph with pip:


```python
%pip install langgraph==0.2.44
%pip install ragas
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
     'timestamp': '2024-11-07T06:35:31.023Z',
     'plan': 'Free',
     'total': 250,
     'used': 13,
     'remaining': 237}




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




    {'gold': 85.3995,
     'silver': 0.9997,
     'platinum': 31.4921,
     'palladium': 33.0797,
     'lbma_gold_am': 87.6494,
     'lbma_gold_pm': 85.5274,
     'lbma_silver': 1.0224,
     'lbma_platinum_am': 31.7328,
     'lbma_platinum_pm': 31.2827,
     'lbma_palladium_am': 33.9833,
     'lbma_palladium_pm': 33.3725,
     'mcx_gold': 90.6623,
     'mcx_gold_am': 92.4892,
     'mcx_gold_pm': 92.5011,
     'mcx_silver': 1.0744,
     'mcx_silver_am': 1.0958,
     'mcx_silver_pm': 1.1044,
     'ibja_gold': 90.7632,
     'copper': 0.0096,
     'aluminum': 0.0026,
     'lead': 0.002,
     'nickel': 0.0161,
     'zinc': 0.003,
     'lme_copper': 0.0094,
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


    
![jpeg](../../_static/imgs/_langgraph_agent_evaluation_28_0.jpg)
    


To test our setup, we will run the agent with a query. The agent will fetch the price of copper using the metals.dev API.


```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="What is the price of copper?")]
result = react_graph.invoke({"messages": messages})
```


```python
result["messages"]
```




    [HumanMessage(content='What is the price of copper?', id='a23dac0e-b7db-4fa3-86bb-35fcee17d40e'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ny714Y9Ql2NKGpBK4WJZI36H', 'function': {'arguments': '{"metal_name":"copper"}', 'name': 'get_metal_price'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 116, 'total_tokens': 134, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-67ef1189-7e0c-4396-9862-3e4a607089bb-0', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'copper'}, 'id': 'call_Ny714Y9Ql2NKGpBK4WJZI36H', 'type': 'tool_call'}], usage_metadata={'input_tokens': 116, 'output_tokens': 18, 'total_tokens': 134}),
     ToolMessage(content='0.0098', name='get_metal_price', id='0f29642b-b4e8-4b78-9923-007b87affbe2', tool_call_id='call_Ny714Y9Ql2NKGpBK4WJZI36H'),
     AIMessage(content='The price of copper is $0.0098 per gram.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 148, 'total_tokens': 162, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'stop', 'logprobs': None}, id='run-71526780-e9f9-4a79-a914-7863f8ec9078-0', usage_metadata={'input_tokens': 148, 'output_tokens': 14, 'total_tokens': 162})]



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

To convert the list of LangChain messages into a format suitable for Ragas evaluation, you have two options:
#### Option 1: Use the Implementation Provided by Ragas
Ragas has already provided a function `convert_to_ragas_messages` that can be used to convert LangChain messages into the format expected by Ragas. Here's how you can use it:


```python
from ragas.integrations.langgraph import convert_to_ragas_messages

# Assuming 'result["messages"]' contains the list of LangChain messages
ragas_trace = convert_to_ragas_messages(result["messages"])
```

#### Option 2: Write a Custom Conversion Function
Alternatively, you can implement your own conversion function by drawing inspiration from the code below. This function converts a list of LangChain messages (HumanMessage, AIMessage, ToolMessage) into Ragas-compatible message objects.


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
convert_to_ragas_trace(messages=result["messages"])  # List of Ragas messages converted from LangChain messages using the defined function
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
messages = [HumanMessage(content="What is the price of 10 grams of silver?")]

result = react_graph.invoke({"messages": messages})
```


```python
result["messages"] # List of Langchain messages
```




    [HumanMessage(content='What is the price of 10 grams of silver?', id='f2839748-85f5-4526-9a5a-33a7863ece80'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Rwg2nNBGcpTckTkPkvuU9BuZ', 'function': {'arguments': '{"metal_name":"silver"}', 'name': 'get_metal_price'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 120, 'total_tokens': 137, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-9e34ecae-0fdf-4d85-b367-3ae291d57121-0', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'silver'}, 'id': 'call_Rwg2nNBGcpTckTkPkvuU9BuZ', 'type': 'tool_call'}], usage_metadata={'input_tokens': 120, 'output_tokens': 17, 'total_tokens': 137}),
     ToolMessage(content='1.0523', name='get_metal_price', id='03e48a10-e762-4476-a429-798d24a18f98', tool_call_id='call_Rwg2nNBGcpTckTkPkvuU9BuZ'),
     AIMessage(content='The current price of silver is approximately $1.0523 per gram. Therefore, the price for 10 grams of silver would be about $10.52.', response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 151, 'total_tokens': 185, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'stop', 'logprobs': None}, id='run-e163708f-5ae4-4fa8-8df1-d4b824baf8d4-0', usage_metadata={'input_tokens': 151, 'output_tokens': 34, 'total_tokens': 185})]




```python
from ragas.integrations.langgraph import convert_to_ragas_messages

ragas_trace = convert_to_ragas_messages(result["messages"]) # List of Ragas messages converted using the Ragas function
```


```python
from ragas.dataset_schema import MultiTurnSample
from ragas.metrics import AgentGoalAccuracyWithReference
from ragas.llms import LangchainLLMWrapper


sample = MultiTurnSample(
    user_input=ragas_trace,
    reference="Price of 10 grams of silver",
)

scorer = AgentGoalAccuracyWithReference()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
scorer.llm = evaluator_llm
await scorer.multi_turn_ascore(sample)
```




    1.0



Agent Goal Accuracy: 1, because the LLM correctly achieved the user’s goal of retrieving the price of 10 grams of silver.

## What’s next
🎉 Congratulations! We have learned how to evaluate an agent using the Ragas evaluation framework.
