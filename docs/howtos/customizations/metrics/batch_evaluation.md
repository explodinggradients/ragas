# Batch Evaluation for Cost Optimization

When running large-scale evaluations, cost can be a significant factor. Ragas now supports OpenAI's Batch API, which offers **up to 50% cost savings** compared to regular API calls, making it ideal for non-urgent evaluation workloads.

## What is Batch Evaluation?

OpenAI's Batch API allows you to submit multiple requests for asynchronous processing at half the cost of synchronous requests. Batch jobs are processed within 24 hours and have separate rate limits, making them perfect for large-scale evaluations where immediate results aren't required.

### Key Benefits

- **50% Cost Savings** on both input and output tokens
- **Higher Rate Limits** that don't interfere with real-time usage
- **Guaranteed Processing** within 24 hours (often much sooner)
- **Large Scale Support** up to 50,000 requests per batch

## Quick Start

### Basic Batch Evaluation

```python
import os
from ragas.batch_evaluation import BatchEvaluator, estimate_batch_cost_savings
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Ensure you have your OpenAI API key set
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Setup LLM with batch support (automatically detected)
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
faithfulness = Faithfulness(llm=llm)

# Prepare your evaluation samples
samples = [
    SingleTurnSample(
        user_input="What is the capital of France?",
        response="The capital of France is Paris.",
        retrieved_contexts=["Paris is the capital city of France."]
    ),
    # ... more samples
]

# Create batch evaluator
evaluator = BatchEvaluator(metrics=[faithfulness])

# Run batch evaluation (blocks until completion)
results = evaluator.evaluate(samples, wait_for_completion=True)

# Check results
for result in results:
    print(f"Metric: {result.metric_name}")
    print(f"Job ID: {result.job_id}")
    print(f"Success Rate: {result.success_rate:.2%}")
    print(f"Sample Count: {result.sample_count}")
```

### Cost Estimation

Before running batch evaluations, you can estimate your cost savings:

```python
from ragas.batch_evaluation import estimate_batch_cost_savings

# Estimate costs for 1000 samples
cost_info = estimate_batch_cost_savings(
    sample_count=1000,
    metrics=[faithfulness],
    regular_cost_per_1k_tokens=0.15,  # GPT-4o-mini input cost
    batch_discount=0.5  # 50% savings
)

print(f"Regular API Cost: ${cost_info['regular_cost']}")
print(f"Batch API Cost: ${cost_info['batch_cost']}")
print(f"Total Savings: ${cost_info['savings']} ({cost_info['savings_percentage']}%)")
```

### Asynchronous Batch Evaluation

For non-blocking operations, use async evaluation:

```python
import asyncio

async def run_batch_evaluation():
    evaluator = BatchEvaluator(metrics=[faithfulness])
    
    # Submit jobs without waiting
    results = await evaluator.aevaluate(
        samples=samples,
        wait_for_completion=False  # Don't block
    )
    
    # Jobs are submitted, check back later
    for result in results:
        print(f"Submitted job {result.job_id} for {result.metric_name}")

# Run async evaluation
asyncio.run(run_batch_evaluation())
```

## Checking Batch Support

Not all LLMs support batch evaluation. Here's how to check:

```python
# Check if metric supports batch evaluation
if faithfulness.supports_batch_evaluation():
    print(f"✅ {faithfulness.name} supports batch evaluation")
else:
    print(f"❌ {faithfulness.name} requires regular API")

# Check LLM batch support
if llm.supports_batch_api():
    print("✅ LLM supports batch processing")
else:
    print("❌ LLM does not support batch processing")
```

## Supported Models

Currently, batch evaluation is supported for:
- OpenAI models (ChatOpenAI, AzureChatOpenAI)
- All metrics that use these LLMs

### Supported Metrics

- ✅ Faithfulness (partial support)
- 🔄 More metrics coming soon...

For metrics not yet supporting batch evaluation, they will automatically fall back to regular API calls.

## Configuration Options

### BatchEvaluator Parameters

```python
evaluator = BatchEvaluator(
    metrics=metrics,
    max_batch_size=1000,        # Max samples per batch
    poll_interval=300.0,        # Status check interval (5 minutes)
    timeout=86400.0            # Max wait time (24 hours)
)
```

### Custom Metadata

Add metadata to track your batch jobs:

```python
results = evaluator.evaluate(
    samples=samples,
    metadata={
        "experiment": "model_comparison",
        "version": "v1.0",
        "dataset": "production_qa"
    }
)
```

## Best Practices

### When to Use Batch Evaluation

✅ **Ideal for:**
- Large-scale evaluations (100+ samples)
- Non-urgent evaluation workloads
- Cost optimization scenarios
- Regular evaluation pipelines

❌ **Avoid for:**
- Real-time evaluation needs
- Interactive applications
- Small datasets (<50 samples)
- Time-sensitive workflows

### Optimization Tips

1. **Batch Size**: Use 1000-5000 samples per batch for optimal performance
2. **Model Selection**: Use cost-effective models like `gpt-4o-mini` 
3. **Concurrent Processing**: Submit multiple metrics simultaneously
4. **Monitoring**: Set up logging for long-running jobs

```python
import logging

# Enable batch evaluation logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ragas.batch_evaluation')
```

## Error Handling

```python
try:
    results = evaluator.evaluate(samples)
    
    for result in results:
        if result.errors:
            print(f"❌ Errors in {result.metric_name}:")
            for error in result.errors:
                print(f"  - {error}")
        else:
            print(f"✅ {result.metric_name}: {result.success_rate:.2%} success")
            
except Exception as e:
    print(f"Batch evaluation failed: {e}")
```

## Low-Level Batch API

For advanced use cases, you can use the low-level batch API directly:

```python
from ragas.llms.batch_api import create_batch_api, BatchRequest
from openai import OpenAI

# Direct batch API usage
client = OpenAI()
batch_api = create_batch_api(client)

# Create custom requests
requests = [
    BatchRequest(
        custom_id="eval-1",
        body={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Evaluate this response..."}]
        }
    )
]

# Submit batch job
batch_job = batch_api.create_batch(requests)
print(f"Batch job created: {batch_job.batch_id}")

# Monitor progress
status = batch_job.get_status()
print(f"Status: {status.value}")

# Retrieve results when complete
if status.value == "completed":
    results = batch_job.get_results()
    for result in results:
        print(f"Response for {result.custom_id}: {result.response}")
```

## Troubleshooting

### Common Issues

**Issue**: "Batch API not supported for this LLM"
```python
# Solution: Use OpenAI-based LLM
from langchain_openai import ChatOpenAI
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
```

**Issue**: "Metric does not support batch evaluation"
```python
# Solution: Check metric support or wait for future updates
if not metric.supports_batch_evaluation():
    print(f"Metric {metric.name} will use regular API")
```

**Issue**: Timeout waiting for batch completion
```python
# Solution: Use non-blocking evaluation or increase timeout
results = evaluator.evaluate(
    samples, 
    wait_for_completion=False  # Don't wait
)
# Or increase timeout
evaluator = BatchEvaluator(timeout=172800.0)  # 48 hours
```

## Migration from Regular Evaluation

Converting existing evaluations to use batch processing is simple:

### Before (Regular API)
```python
from ragas import evaluate
from ragas.metrics import Faithfulness

results = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(llm=llm)]
)
```

### After (Batch API)
```python
from ragas.batch_evaluation import BatchEvaluator
from ragas.metrics import Faithfulness

# Convert dataset to samples if needed
samples = [sample for sample in eval_dataset]

evaluator = BatchEvaluator(metrics=[Faithfulness(llm=llm)])
results = evaluator.evaluate(samples)
```

The batch API provides significant cost savings while maintaining the same evaluation quality, making it an excellent choice for large-scale evaluation workloads.