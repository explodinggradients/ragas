# Async Evaluation

Ragas provides first-class async support for server applications and async frameworks through the `aevaluate()` function, addressing compatibility issues with `nest_asyncio` in production environments.

## The Problem

Previously, Ragas would automatically call `nest_asyncio.apply()` during evaluation, which caused problems in server environments:

- Broke FastAPI servers and other async applications
- Caused "asyncio.run() cannot be called from a running event loop" errors  
- Forced unwanted modifications to the global event loop

## The Solution: aevaluate()

The new `aevaluate()` function provides native async support without `nest_asyncio`:

```python
import asyncio
from ragas import aevaluate

async def main():
    result = await aevaluate(dataset, metrics=metrics)
    print(result)

asyncio.run(main())
```

## Usage Examples

### FastAPI Server

```python
from fastapi import FastAPI
from ragas import aevaluate
from ragas.dataset_schema import EvaluationDataset

app = FastAPI()

@app.post("/evaluate")
async def evaluate_endpoint(dataset_data: dict):
    # Convert your data to EvaluationDataset
    dataset = EvaluationDataset.from_list(dataset_data["samples"])
    
    # Use aevaluate for async evaluation
    result = await aevaluate(
        dataset=dataset,
        metrics=your_metrics
    )
    
    return {"evaluation_result": result.to_pandas().to_dict()}
```

### Multiple Concurrent Evaluations

```python
import asyncio
from ragas import aevaluate

async def evaluate_multiple_datasets(datasets, metrics):
    """Evaluate multiple datasets concurrently."""
    tasks = []
    
    for i, dataset in enumerate(datasets):
        task = aevaluate(
            dataset=dataset, 
            metrics=metrics,
            experiment_name=f"evaluation_{i}"
        )
        tasks.append(task)
    
    # Run all evaluations concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

# Usage
datasets = [dataset1, dataset2, dataset3]
results = asyncio.run(evaluate_multiple_datasets(datasets, metrics))
```

### Django Async Views

```python
from django.http import JsonResponse
from ragas import aevaluate
import asyncio

async def evaluate_view(request):
    """Django async view using aevaluate."""
    if request.method == 'POST':
        # Process request data
        dataset = process_request_data(request.POST)
        
        # Async evaluation
        result = await aevaluate(dataset, metrics=metrics)
        
        return JsonResponse({
            'status': 'success',
            'scores': result.to_pandas().to_dict()
        })
```

### Async Context Manager

```python
import asyncio
from ragas import aevaluate
from contextlib import asynccontextmanager

@asynccontextmanager
async def evaluation_session(config):
    """Async context manager for evaluation sessions."""
    print(f"Starting evaluation session with config: {config}")
    try:
        yield config
    finally:
        print("Evaluation session completed")

async def run_evaluation_session():
    config = {"batch_size": 10, "show_progress": True}
    
    async with evaluation_session(config) as session_config:
        result = await aevaluate(
            dataset=dataset,
            metrics=metrics,
            batch_size=session_config["batch_size"],
            show_progress=session_config["show_progress"]
        )
    
    return result
```

## API Compatibility

Both `evaluate()` and `aevaluate()` have identical parameters (except `return_executor`):

```python
# Sync version
result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
    experiment_name="my_experiment",
    show_progress=True,
    batch_size=10
)

# Async version - same parameters
result = await aevaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
    experiment_name="my_experiment",
    show_progress=True,
    batch_size=10
)
```

## Error Handling

The improved `evaluate()` function now provides helpful error messages when used incorrectly:

```python
import asyncio

async def wrong_usage():
    # This will raise a clear error message
    try:
        result = evaluate(dataset, metrics=metrics)  # Wrong in async context
    except RuntimeError as e:
        print(e)
        # Error: Cannot run sync evaluate() from within an async context...
        # Use 'aevaluate()' instead for proper async support

    # Correct usage in async context
    result = await aevaluate(dataset, metrics=metrics)
    return result

asyncio.run(wrong_usage())
```

## Performance Benefits

Using `aevaluate()` in async applications provides several benefits:

1. **No Event Loop Conflicts**: Works properly in server environments
2. **Better Concurrency**: True async evaluation without blocking
3. **Resource Efficiency**: Better memory and CPU utilization  
4. **Scalability**: Handle multiple concurrent evaluations

## Migration Guide

### For Server Applications

**Before (problematic):**
```python
def evaluate_handler(dataset):
    # This would break servers
    return evaluate(dataset, metrics=metrics)
```

**After (fixed):**
```python
async def evaluate_handler(dataset):
    # Works perfectly in async contexts
    return await aevaluate(dataset, metrics=metrics)
```

### For Existing Sync Code

No changes needed for regular Python scripts:

```python
from ragas import evaluate

# Still works as before
result = evaluate(dataset, metrics=metrics)
```

## Context Detection

The `evaluate()` function automatically detects the execution context:

- **Regular Python scripts**: Uses `asyncio.run()` normally
- **Jupyter notebooks**: Applies `nest_asyncio` safely  
- **Server contexts**: Raises helpful error directing to `aevaluate()`

## Best Practices

1. **Use `aevaluate()` in async applications** (FastAPI, Django async, etc.)
2. **Use `evaluate()` in sync scripts** and notebooks
3. **Handle exceptions properly** in async code
4. **Use appropriate batch sizes** for concurrent evaluations
5. **Monitor resource usage** when running multiple evaluations

## Troubleshooting

### Common Issues

**"Cannot run sync evaluate() from within an async context"**
- Solution: Use `aevaluate()` instead of `evaluate()`

**"asyncio.run() cannot be called from a running event loop"**  
- This should no longer occur with `aevaluate()`
- If it does, ensure you're using `await aevaluate()` not `evaluate()`

**Import errors**
```python
# Make sure aevaluate is available
from ragas import aevaluate, evaluate
```

### Performance Tips

- Use `batch_size` parameter to control memory usage
- Set `show_progress=False` in server environments
- Consider using `asyncio.gather()` for concurrent evaluations
- Monitor async task counts to avoid overwhelming the system