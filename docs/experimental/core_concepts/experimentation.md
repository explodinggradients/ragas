# Experiments

## What is an experiment?

An experiment is a deliberate change made to your application to test a hypothesis or idea. For example, in a Retrieval-Augmented Generation (RAG) system, you might replace the retriever model to evaluate how a new embedding model impacts chatbot responses.

### Principles of a Good Experiment

1. **Define measurable metrics**: Use metrics like accuracy, precision, or recall to quantify the impact of your changes.
2. **Systematic result storage**: Ensure results are stored in an organized manner for easy comparison and tracking.
3. **Isolate changes**: Make one change at a time to identify its specific impact. Avoid making multiple changes simultaneously, as this can obscure the results.
4. **Iterative process**: Follow a structured approach: *Make a change → Run evaluations → Observe results →
```mermaid
graph LR
    A[Make a change] --> B[Run evaluations]
    B --> C[Observe results]
    C --> D[Hypothesize next change]
    D --> A
```

## Experiments in Ragas

### Components of an Experiment

1. **Test dataset**: The data used to evaluate the system.
2. **Application endpoint**: The application, component or model being tested.
3. **Metrics**: Quantitative measures to assess performance.

### Execution Process

Running an experiment involves:

1. Executing the dataset against the application endpoint.
2. Calculating metrics to quantify performance.
3. Returning and storing the results.

## Using the `@experiment` Decorator

The `@experiment` decorator in Ragas simplifies the orchestration, scaling, and storage of experiments. Here's an example:

```python
from ragas import experiment

# Define your metric and dataset
my_metric = ...
dataset = ...

@experiment
async def my_experiment(row):
    # Process the query through your application
    response = my_app(row.query)
    
    # Calculate the metric
    metric = my_metric.score(response, row.ground_truth)
    
    # Return results
    return {**row, "response": response, "accuracy": metric.value}

# Run the experiment
my_experiment.arun(dataset)
```

### Passing Additional Parameters

You can pass additional parameters to your experiment function through `arun()`. This is useful for models, configurations, or any other parameters your experiment needs:

```python
@experiment
async def my_experiment(row, model):
    # Process the query with the specified parameters
    response = my_app(row.query, model=model)
    
    # Calculate the metric
    metric = my_metric.score(response, row.ground_truth)
    
    # Return results
    return {**row, "response": response, "accuracy": metric.value}

# Run with specific parameters
my_experiment.arun(dataset, "gpt-4")

# Or use keyword arguments
my_experiment.arun(dataset, model="gpt-4o")
```

### Using Data Models

You can specify a data model for your experiment results at the decorator level:

```python
from pydantic import BaseModel

class ExperimentResult(BaseModel):
    response: str
    accuracy: float
    model_used: str

@experiment(experiment_model=ExperimentResult)
async def my_experiment(row, model):
    response = my_app(row.query, model)
    metric = my_metric.score(response, row.ground_truth)
    return ExperimentResult(
        response=response, 
        accuracy=metric.value, 
        model_used=model
    )

# Run experiment with specific model
my_experiment.arun(dataset, model="gpt-4o")
```

### Complete Example: LLM Parameter Passing

Here's a complete example showing how to pass different LLM models to your experiment function:

```python
from pydantic import BaseModel
from ragas.experimental import Dataset
from ragas import experiment

class ExperimentResult(BaseModel):
    query: str
    response: str
    accuracy: float
    model_used: str
    latency_ms: float

@experiment(experiment_model=ExperimentResult)
async def llm_experiment(row, llm_model, temperature=0.7):
    """Experiment function that accepts LLM model and other parameters."""
    import time
    start_time = time.time()
    
    # Use the passed LLM model
    response = await my_llm_app(
        query=row.query, 
        model=llm_model,
        temperature=temperature
    )
    
    # Calculate metrics
    metric = my_metric.score(response, row.ground_truth)
    end_time = time.time()
    
    return ExperimentResult(
        query=row.query,
        response=response,
        accuracy=metric.value,
        model_used=llm_model,
        latency_ms=(end_time - start_time) * 1000
    )

# Run experiments with different models
gpt4_results = await llm_experiment.arun(
    dataset, 
    llm_model="gpt-4o",
    temperature=0.1
)

claude_results = await llm_experiment.arun(
    dataset,
    llm_model="claude-4-sonnet",
    temperature=0.7
)
```

## Result Storage

Once executed, Ragas processes each row in the dataset, runs it through the function, and stores the results in the `experiments` folder. The storage backend can be configured based on your preferences.