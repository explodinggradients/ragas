# Metrics for evaluating AI Applications

## Types of Metrics in AI Applications

1. **End-to-End Metrics**

End-to-end metrics evaluate the overall system performance from the user’s perspective, treating the AI application as a black box. These metrics quantify key outcomes users care deeply about, based solely on the system’s final outputs.

Examples:
	• Answer correctness: Measures if the provided answers from a Retrieval-Augmented Generation (RAG) system are accurate.
	• Citation accuracy: Evaluates whether the references cited by the RAG system are correctly identified and relevant.

Optimizing end-to-end metrics ensures tangible improvements aligned directly with user expectations.

2. **Component-Level Metrics**

Component-level metrics assess the individual parts of an AI system independently. These metrics are immediately actionable and facilitate targeted improvements but do not necessarily correlate directly with end-user satisfaction.

Example:
	• Retrieval accuracy: Measures how effectively a RAG system retrieves relevant information. A low retrieval accuracy (e.g., 50%) signals that improving this component can enhance overall system performance. However, improving a component alone doesn’t guarantee better end-to-end outcomes.

3. **Business Metrics**

Business metrics align AI system performance with organizational objectives and quantify tangible business outcomes. These metrics are typically lagging indicators, calculated after a deployment period (days/weeks/months).

Example:
	•	Ticket deflection rate: Measures the percentage reduction of support tickets due to the deployment of an AI assistant.

## Choosing the Right Metrics for Your Application

1. **Prioritize End-to-End Metrics**

Focus first on metrics reflecting overall user satisfaction. While many aspects influence user satisfaction—such as factual correctness, response tone, and explanation depth—concentrate initially on the few dimensions delivering maximum user value (e.g., answer and citation accuracy in a RAG-based assistant).

2. **Ensure Interpretability**

Design metrics clear enough for the entire team to interpret and reason about. For example:
	• Execution accuracy in a text-to-SQL system: Does the SQL query generated return precisely the same dataset as the ground truth query crafted by domain experts?

3. **Emphasize Objective Over Subjective Metrics**

Prioritize metrics with objective criteria, minimizing subjective judgment. Assess objectivity by independently labeling samples across team members and measuring agreement levels. A high inter-rater agreement (≥80%) indicates greater objectivity.

4. **Few Strong Signals over Many Weak Signals**

Avoid a proliferation of metrics that provide weak signals and impede clear decision-making. Instead, select fewer metrics offering strong, reliable signals. For instance:
	• In a conversational AI, goal accuracy (whether the user’s objective was met) provides clearer and more actionable insights than subjective measures like coherence or helpfulness.

## LLM-based vs. Non-LLM-based Metrics

### Non-LLM-based Metrics

These metrics are deterministic functions evaluating predefined inputs against clear, finite criteria.

Example:

```python
def my_metric(predicted: str, expected: str) -> str:
    return "pass" if predicted.lower() == expected.lower() else "fail"
```

When to use:
	• Tasks with strictly defined correct outcomes (e.g., mathematical solutions, deterministic tasks like booking agents updating databases).

### LLM-based Metrics

These leverage LLMs (Large Language Models) to evaluate outcomes, typically useful where correctness is nuanced or highly variable.

Example:
```python
def my_metric(predicted: str, expected: str) -> str:
    response = llm.generate(f"Evaluate semantic similarity between '{predicted}' and '{expected}'")
    return "pass" if response > 5 else "fail"
```

When to use:
	• Tasks with numerous valid outcomes (e.g., paraphrased correct answers).
	• Complex evaluation criteria aligned with human or expert preferences (e.g., distinguishing “deep” vs. “shallow” insights in research reports). Although simpler metrics (length or keyword count) are possible, LLM-based metrics capture nuanced human judgment more effectively.