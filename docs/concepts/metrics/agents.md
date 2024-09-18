# Agentic or tool use metrics

Agentic or tool use workflows can be evaluated in multiple dimensions. Here are some of the metrics that can be used to evaluate the performance of agents or tools in a given task.

## Tool call Accuracy

Tool call accuracy is a metric that can be used to evaluate the performance of the LLM in identifying and calling the required tools to complete a given task. This metric needs `user_input` and `reference_tool_calls` to evaluate the performance of the LLM in identifying and calling the required tools to complete a given task. The metric is computed by comparing the `reference_tool_calls` with the Tool calls made by the AI. The values range between 0 and 1, with higher values indicating better performance. 

```{code-block} python
from ragas.dataset_schema import  MultiTurnSample
from ragas.messages import HumanMessage,AIMessage,ToolMessage,ToolCall
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy


sample = MultiTurnSample(user_input=[
    HumanMessage(content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"),
    AIMessage(content="Sure, let me find the best options for you.", tool_calls=[
        ToolCall(name="restaurant_search", args={"cuisine": "Asian", "time": "8:00pm"})
    ]),
              ],
    reference_tool_calls=[ToolCall(name="restaurant_book", args={"name": "Golden", "time": "8:00pm"})
])

scorer = ToolCallAccuracy()
await metric.multi_turn_ascore(sample)
```

By default the tool names and arguments are compared using exact string matching. But sometimes this might not be optimal, for example if the args are natural language strings. You can also use any ragas metrics (values between 0 and 1) as distance measure to identify if a retrieved context is relevant or not. For example,

```{code-block} python
from ragas.metrics._string import NonLLMStringSimilarity
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy

metric = ToolCallAccuracy()
metric.arg_comparison_metric = NonLLMStringSimilarity()
```

## Agent Goal accuracy


Agent goal accuracy is a metric that can be used to evaluate the performance of the LLM in identifying and achieving the goals of the user. This is a binary metric, with 1 indicating that the AI has achieved the goal and 0 indicating that the AI has not achieved the goal.

### With reference

Calculating agent goal accuracy with reference needs `user_input` and `reference` to evaluate the performance of the LLM in identifying and achieving the goals of the user. The annotated `reference` will be used as ideal outcome. The metric is computed by comparing the `reference` with the goal achieved by the end of workflow.


```{code-block} python
from ragas.dataset_schema import  MultiTurnSample
from ragas.messages import HumanMessage,AIMessage,ToolMessage,ToolCall
from ragas.metrics._agent_goal_accuracy import AgentGoalAccuracyWithReference


sample = MultiTurnSample(user_input=[
    HumanMessage(content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"),
    AIMessage(content="Sure, let me find the best options for you.", tool_calls=[
        ToolCall(name="restaurant_search", args={"cuisine": "Chinese", "time": "8:00pm"})
    ]),
    ToolMessage(content="Found a few options: 1. Golden Dragon, 2. Jade Palace"),
    AIMessage(content="I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?"),
    HumanMessage(content="Let's go with Golden Dragon."),
    AIMessage(content="Great choice! I'll book a table for 8:00pm at Golden Dragon.", tool_calls=[
        ToolCall(name="restaurant_book", args={"name": "Golden Dragon", "time": "8:00pm"})
    ]),
    ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),
    AIMessage(content="Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!"),
    HumanMessage(content="thanks"),
],
    reference="Table booked at one of the chinese restaurants at 8 pm")

scorer = AgentGoalAccuracyWithReference()
await metric.multi_turn_ascore(sample)

```

### Without reference

In without reference mode, the metric will evaluate the performance of the LLM in identifying and achieving the goals of the user without any reference. Here the desired outcome is inferred from the human interactions in the workflow.


```{code-block} python


```{code-block} python
from ragas.dataset_schema import  MultiTurnSample
from ragas.messages import HumanMessage,AIMessage,ToolMessage,ToolCall
from ragas.metrics._agent_goal_accuracy import AgentGoalAccuracyWithoutReference


sample = MultiTurnSample(user_input=[
    HumanMessage(content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"),
    AIMessage(content="Sure, let me find the best options for you.", tool_calls=[
        ToolCall(name="restaurant_search", args={"cuisine": "Chinese", "time": "8:00pm"})
    ]),
    ToolMessage(content="Found a few options: 1. Golden Dragon, 2. Jade Palace"),
    AIMessage(content="I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?"),
    HumanMessage(content="Let's go with Golden Dragon."),
    AIMessage(content="Great choice! I'll book a table for 8:00pm at Golden Dragon.", tool_calls=[
        ToolCall(name="restaurant_book", args={"name": "Golden Dragon", "time": "8:00pm"})
    ]),
    ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),
    AIMessage(content="Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!"),
    HumanMessage(content="thanks"),
])

scorer = AgentGoalAccuracyWithoutReference()
await metric.multi_turn_ascore(sample)

```
