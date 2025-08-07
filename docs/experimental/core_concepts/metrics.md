# Metrics

## Why Metrics Matter

You can't improve what you don't measure. Metrics are the feedback loop that makes iteration possible.

In AI systems, progress depends on running many experiments—each a hypothesis about how to improve performance. But without a clear, reliable metric, you can't tell the difference between a successful experiment (a positive delta between the new score and the old one) and a failed one.

Metrics give you a compass. They let you quantify improvement, detect regressions, and align optimization efforts with user impact and business value.

## Types of Metrics in AI Applications

### 1. End-to-End Metrics

End-to-end metrics evaluate the overall system performance from the user's perspective, treating the AI application as a black box. These metrics quantify key outcomes users care deeply about, based solely on the system's final outputs.

Examples:

- Answer correctness: Measures if the provided answers from a Retrieval-Augmented Generation (RAG) system are accurate.
- Citation accuracy: Evaluates whether the references cited by the RAG system are correctly identified and relevant.

Optimizing end-to-end metrics ensures tangible improvements aligned directly with user expectations.

### 2. Component-Level Metrics

Component-level metrics assess the individual parts of an AI system independently. These metrics are immediately actionable and facilitate targeted improvements but do not necessarily correlate directly with end-user satisfaction.

Example:

- Retrieval accuracy: Measures how effectively a RAG system retrieves relevant information. A low retrieval accuracy (e.g., 50%) signals that improving this component can enhance overall system performance. However, improving a component alone doesn't guarantee better end-to-end outcomes.

### 3. Business Metrics

Business metrics align AI system performance with organizational objectives and quantify tangible business outcomes. These metrics are typically lagging indicators, calculated after a deployment period (days/weeks/months).

Example:

- Ticket deflection rate: Measures the percentage reduction of support tickets due to the deployment of an AI assistant.

## Types of Metrics in Ragas

In Ragas, we categorize metrics based on the type of output they produce. This classification helps clarify how each metric behaves and how its results can be interpreted or aggregated. The three types are:

### 1. Discrete Metrics

These return a single value from a predefined list of categorical classes. There is no implicit ordering among the classes. Common use cases include classifying outputs into categories such as pass/fail or good/okay/bad.

Example:
```python
from ragas_experimental.metrics import discrete_metric

@discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
def my_metric(predicted: str, expected: str) -> str:
    return "pass" if predicted.lower() == expected.lower() else "fail"

```

### 2. Numeric Metrics

These return an integer or float value within a specified range. Numeric metrics support aggregation functions such as mean, sum, or mode, making them useful for statistical analysis.
    
```python
from ragas_experimental.metrics import numeric_metric

@numeric_metric(name="response_accuracy", allowed_values=(0, 1))
def my_metric(predicted: float, expected: float) -> float:
    return abs(predicted - expected) / max(expected, 1e-5)

my_metric.score(predicted=0.8, expected=1.0)  # Returns a float value
```

### 3. Ranking Metrics

These evaluate multiple outputs at once and return a ranked list based on a defined criterion. They are useful when the goal is to compare multiple outputs from the same pipeline relative to one another.

```python
from ragas_experimental.metrics import ranked_metric
@ranked_metric(name="response_ranking", allowed_values=[0,1])
def my_metric(responses: list) -> list:
    response_lengths = [len(response) for response in responses]
    sorted_indices = sorted(range(len(response_lengths)), key=lambda i: response_lengths[i])
    return sorted_indices

my_metric.score(responses=["short", "a bit longer", "the longest response"])  # Returns a ranked list of indices
```

## LLM-based vs. Non-LLM-based Metrics

### Non-LLM-based Metrics

These metrics are deterministic functions evaluating predefined inputs against clear, finite criteria.

Example:

```python
def my_metric(predicted: str, expected: str) -> str:
    return "pass" if predicted.lower() == expected.lower() else "fail"
```

When to use:

- Tasks with strictly defined correct outcomes (e.g., mathematical solutions, deterministic tasks like booking agents updating databases).

### LLM-based Metrics

These leverage LLMs (Large Language Models) to evaluate outcomes, typically useful where correctness is nuanced or highly variable.

Example:
```python
from ragas_experimental.metrics import DiscreteMetric

my_metric = DiscreteMetric(
    name="response_quality",
    prompt="Evaluate the response based on the pass criteria: {pass_criteria}. Does the response meet the criteria? Return 'pass' or 'fail'.\nResponse: {response}",
    allowed_values=["pass", "fail"]
)
```

When to use:

- Tasks with numerous valid outcomes (e.g., paraphrased correct answers).
- Complex evaluation criteria aligned with human or expert preferences (e.g., distinguishing "deep" vs. "shallow" insights in research reports). Although simpler metrics (length or keyword count) are possible, LLM-based metrics capture nuanced human judgment more effectively.

## Choosing the Right Metrics for Your Application

### 1. Prioritize End-to-End Metrics

Focus first on metrics reflecting overall user satisfaction. While many aspects influence user satisfaction—such as factual correctness, response tone, and explanation depth—concentrate initially on the few dimensions delivering maximum user value (e.g., answer and citation accuracy in a RAG-based assistant).

### 2. Ensure Interpretability

Design metrics clear enough for the entire team to interpret and reason about. For example:

- Execution accuracy in a text-to-SQL system: Does the SQL query generated return precisely the same dataset as the ground truth query crafted by domain experts?

### 3. Emphasize Objective Over Subjective Metrics

Prioritize metrics with objective criteria, minimizing subjective judgment. Assess objectivity by independently labeling samples across team members and measuring agreement levels. A high inter-rater agreement (≥80%) indicates greater objectivity.

### 4. Few Strong Signals over Many Weak Signals

Avoid a proliferation of metrics that provide weak signals and impede clear decision-making. Instead, select fewer metrics offering strong, reliable signals. For instance:

- In a conversational AI, using a single metric such as goal accuracy (whether the user's objective for interacting with the AI was met) provides strong proxy for the performance of the system than multiple weak proxies like coherence or helpfulness.
