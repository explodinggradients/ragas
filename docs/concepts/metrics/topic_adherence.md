## Topic Adherence

AI systems deployed in real-world applications are expected to adhere to domains of interest while interacting with users but LLMs sometimes may answer general queries by ignoring this limitation. The topic adherence metric evaluates the ability of the AI to stay on predefined domains during the interactions. This metric is particularly important in conversational AI systems, where the AI is expected to only provide assistance to queries related to predefined domains.

Topic adherence requires a predefined set of topics that the AI system is expected to adhere to which is provided using `reference_topics` along with `user_input`. The metric can compute precision, recall, and F1 score for topic adherence, defined as 
    
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
from ragas.metrics._topic_adherence import TopicAdherenceScore


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

