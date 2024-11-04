# Agentic or Tool use

Agentic or tool use workflows can be evaluated in multiple dimensions. Here are some of the metrics that can be used to evaluate the performance of agents or tools in a given task.


## Topic Adherence

AI systems deployed in real-world applications are expected to adhere to domains of interest while interacting with users but LLMs sometimes may answer general queries by ignoring this limitation. The topic adherence metric evaluates the ability of the AI to stay on predefined domains during the interactions. This metric is particularly important in conversational AI systems, where the AI is expected to only provide assistance to queries related to predefined domains.

`TopicAdherenceScore` requires a predefined set of topics that the AI system is expected to adhere to which is provided using `reference_topics` along with `user_input`. The metric can compute precision, recall, and F1 score for topic adherence, defined as 
    
$$
\text{Precision } = {|\text{Queries that are answered and are adheres to any present reference topics}| \over |\text{Queries that are answered and are adheres to any present reference topics}| + |\text{Queries that are answered and do not adheres to any present reference topics}|}
$$

$$
\text{Recall } = {|\text{Queries that are answered and are adheres to any present reference topics}| \over |\text{Queries that are answered and are adheres to any present reference topics}| + |\text{Queries that were refused and should have been answered}|}
$$

$$
\text{F1 Score } = {2 \times \text{Precision} \times \text{Recall} \over \text{Precision} + \text{Recall}}
$$

### Example

```python
from ragas.dataset_schema import  SingleTurnSample, MultiTurnSample, EvaluationDataset
from ragas.messages import HumanMessage,AIMessage,ToolMessage,ToolCall
from ragas.metrics import TopicAdherenceScore


sample_input_4 = [
HumanMessage(content="Can you provide me with details about Einstein's theory of relativity?"),
AIMessage(content="Sure, let me retrieve the relevant information for you.", tool_calls=[
    ToolCall(name="document_search", args={"query": "Einstein's theory of relativity"})
]),
ToolMessage(content="Found relevant documents: 1. Relativity: The Special and the General Theory, 2. General Theory of Relativity by A. Einstein."),
AIMessage(content="I found some documents on Einstein's theory of relativity. Which one would you like to know more about: 'Relativity: The Special and the General Theory' or 'General Theory of Relativity by A. Einstein'?"),
HumanMessage(content="Tell me about the 'General Theory of Relativity'."),
AIMessage(content="Got it! Let me fetch more details from 'General Theory of Relativity by A. Einstein'.", tool_calls=[
    ToolCall(name="document_retrieve", args={"document": "General Theory of Relativity by A. Einstein"})
]),
ToolMessage(content="The document discusses how gravity affects the fabric of spacetime, describing the relationship between mass and spacetime curvature."),
AIMessage(content="The 'General Theory of Relativity' explains how gravity affects the fabric of spacetime and the relationship between mass and spacetime curvature. Would you like more details or a specific explanation?"),
HumanMessage(content="No, that's perfect. By the way, do you know any good recipes for a chocolate cake?"),
AIMessage(content="Sure! Let me find a simple and delicious recipe for a chocolate cake.", tool_calls=[
    ToolCall(name="recipe_search", args={"query": "chocolate cake recipe"})
]),
ToolMessage(content="Here’s a popular recipe for a chocolate cake: Ingredients include flour, sugar, cocoa powder, eggs, milk, and butter. Instructions: Mix dry ingredients, add wet ingredients, and bake at 350°F for 30-35 minutes."),
AIMessage(content="I found a great recipe for chocolate cake! Would you like the full details, or is that summary enough?")
]


sample = MultiTurnSample(user_input=sample_input_4, reference_topics=["science"])
scorer = TopicAdherenceScore(mode="precision")
scorer.llm = openai_model
await scorer.multi_turn_ascore(sample)
```


To change the mode to recall, set the `mode` parameter to `recall`.

```python
scorer = TopicAdherenceScore(mode="recall")
```  



## Tool call Accuracy

`ToolCallAccuracy` is a metric that can be used to evaluate the performance of the LLM in identifying and calling the required tools to complete a given task. This metric needs `user_input` and `reference_tool_calls` to evaluate the performance of the LLM in identifying and calling the required tools to complete a given task. The metric is computed by comparing the `reference_tool_calls` with the Tool calls made by the AI. The values range between 0 and 1, with higher values indicating better performance. 

```python
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import  MultiTurnSample
from ragas.messages import HumanMessage,AIMessage,ToolMessage,ToolCall

sample = [
    HumanMessage(content="What's the weather like in New York right now?"),
    AIMessage(content="The current temperature in New York is 75°F and it's partly cloudy.", tool_calls=[
        ToolCall(name="weather_check", args={"location": "New York"})
    ]),
    HumanMessage(content="Can you translate that to Celsius?"),
    AIMessage(content="Let me convert that to Celsius for you.", tool_calls=[
        ToolCall(name="temperature_conversion", args={"temperature_fahrenheit": 75})
    ]),
    ToolMessage(content="75°F is approximately 23.9°C."),
    AIMessage(content="75°F is approximately 23.9°C.")
]

sample = MultiTurnSample(
    user_input=sample,
    reference_tool_calls=[
        ToolCall(name="weather_check", args={"location": "New York"}),
        ToolCall(name="temperature_conversion", args={"temperature_fahrenheit": 75})
    ]
)

scorer = ToolCallAccuracy()
scorer.llm = your_llm
await scorer.multi_turn_ascore(sample)
```

The tool call sequence specified in `reference_tool_calls` is used as the ideal outcome. If the tool calls made by the AI does not the the order or sequence of the `reference_tool_calls`, the metric will return a score of 0. This helps to ensure that the AI is able to identify and call the required tools in the correct order to complete a given task.

By default the tool names and arguments are compared using exact string matching. But sometimes this might not be optimal, for example if the args are natural language strings. You can also use any ragas metrics (values between 0 and 1) as distance measure to identify if a retrieved context is relevant or not. For example,

```python
from ragas.metrics._string import NonLLMStringSimilarity
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy

metric = ToolCallAccuracy()
metric.arg_comparison_metric = NonLLMStringSimilarity()
```

## Agent Goal accuracy


Agent goal accuracy is a metric that can be used to evaluate the performance of the LLM in identifying and achieving the goals of the user. This is a binary metric, with 1 indicating that the AI has achieved the goal and 0 indicating that the AI has not achieved the goal.

### With reference

Calculating `AgentGoalAccuracyWithReference` with reference needs `user_input` and `reference` to evaluate the performance of the LLM in identifying and achieving the goals of the user. The annotated `reference` will be used as ideal outcome. The metric is computed by comparing the `reference` with the goal achieved by the end of workflow.


```python
from ragas.dataset_schema import  MultiTurnSample
from ragas.messages import HumanMessage,AIMessage,ToolMessage,ToolCall
from ragas.metrics import AgentGoalAccuracyWithReference


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
scorer.llm = your_llm
await scorer.multi_turn_ascore(sample)

```

### Without reference

`AgentGoalAccuracyWithoutReference` works in without reference mode, the metric will evaluate the performance of the LLM in identifying and achieving the goals of the user without any reference. Here the desired outcome is inferred from the human interactions in the workflow.


### Example

```python
from ragas.dataset_schema import  MultiTurnSample
from ragas.messages import HumanMessage,AIMessage,ToolMessage,ToolCall
from ragas.metrics import AgentGoalAccuracyWithoutReference


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
