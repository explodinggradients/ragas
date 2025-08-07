# How to Evaluate a New LLM For Your Use Case

When a new LLM is released, you might want to determine if it outperforms your current model for your specific use case. This guide shows you how to run an accuracy comparison between two models using Ragas experimental framework.

## What you'll accomplish

By the end of this guide, you'll have:

- Set up a structured evaluation comparing two LLMs
- Evaluated model performance on a realistic business task
- Generated detailed results to inform your model selection decision

## Prerequisites

- Python environment with Ragas experimental installed
- OpenAI API key (or access to your preferred LLM provider)
- Basic familiarity with LLMs

### Installing dependencies

1. Install ragas_examples 
```bash
pip install ragas_experimental[examples]
```
2. Setup your OpenAI API key
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## The evaluation scenario

We'll use eligibility reasoning as our test case: given a customer profile, determine if they qualify for a discount and explain why. This task requires rule-chaining and explanation - skills that differentiate model capabilities.

*Note: You can adapt this approach to any use case that matters for your application.*

> **💡 Quick Start**: If you want to see the complete evaluation in action, you can jump straight to the [end-to-end command](#running-the-evaluation-end-to-end) that runs everything and generates comparison results automatically.
> 
> **📁 Full Code**: The complete source code for this example is available at: https://github.com/explodinggradients/ragas/tree/main/experimental/ragas_examples/benchmark_llm

## Step 1: Set up your environment and API access

First, ensure you have your API credentials configured:

```bash
export OPENAI_API_KEY=your_actual_api_key
```

## Step 2: Configure the models to compare

To customize the evaluation, create a local directory and copy the configuration files:

```bash
mkdir my_llm_evaluation
cd my_llm_evaluation
```

Create a `config.py` file to specify which models you want to evaluation:

```python
# Model configuration for evaluationing
BASELINE_MODEL = "gpt-4.1-mini"        # Your current model
CANDIDATE_MODEL = "o4-mini"  # The new model to evaluate
```

The baseline model represents your current choice, while the candidate is the new model you're considering.

## Step 3: Examine the evaluation dataset

The evaluation uses a pre-built dataset with eligibility reasoning test cases that includes:

- Simple cases with clear outcomes
- Edge cases at rule boundaries  
- Complex scenarios with ambiguous information

Each case specifies:

- `customer_profile`: The input data
- `expected_eligible`: Whether discount applies (True/False)
- `expected_discount`: Expected discount percentage
- `description`: Case complexity indicator

To customize the dataset for your use case, create a `datasets/` directory and add your own CSV file following the same structure. You can also use other formats. Refer to [Datasets - Core Concepts](../core_concepts/datasets.md) for more information. 

## Step 4: Run the evaluation evaluation

For a quick test with default settings, run:

```bash
python -m ragas_examples.benchmark_llm.evals
```

For custom configurations, you'll need to copy and modify the evaluation code locally. The key components are:

- Model configuration (`config.py`)
- Prompt definition (`prompt.py`) 
- Evaluation logic (`evals.py`)
- Test dataset (`datasets/eligibility_evaluation.csv`)

The system will:

1. Load your test dataset
2. Run each case through both models  
3. Parse and score the responses
4. Generate detailed comparison results

## Step 5: Interpret results and make your decision

### Analyze the accuracy metrics

The evaluation provides three key numbers:
- **Baseline accuracy**: How well your current model performs
- **Candidate accuracy**: How well the new model performs  
- **Performance difference**: The gap between them

### Review detailed case results

Examine the detailed case-by-case breakdown to understand:
- Which types of problems each model struggles with
- Whether failures occur on simple or complex cases
- Consistency of performance across different scenarios

### Consider additional factors

While accuracy is crucial, also evaluate:
- **Cost**: Token pricing differences between models
- **Latency**: Response time requirements for your application
- **Consistency**: Reliability of structured output format

*In production systems, these factors often influence the final decision as much as raw accuracy.*

### Make your selection decision

Choose the candidate model if:
- It significantly outperforms the baseline (>5% improvement)
- The performance gain justifies any cost/latency tradeoffs
- It handles your most critical use cases reliably

Stick with your baseline if:
- Performance differences are minimal (<2%)
- The candidate model fails on cases critical to your business
- Cost or latency constraints favor the current model

## Adapting to your use case

To evaluate models for your specific application, you'll need to create your own local implementation:

1. **Copy the example structure**: Extract or recreate the evaluation files locally
2. **Replace the prompt**: Modify the system prompt and output format for your task
3. **Update the dataset**: Create test cases relevant to your domain in CSV format
4. **Adjust the metric**: Update the evaluation logic to match your success criteria
5. **Configure models**: Specify which models you want to compare

The Ragas experimental framework handles the orchestration, parallel execution, and result aggregation automatically.

## Running the evaluation end-to-end

If you want to run the complete evaluation with default settings:

1. Setup your OpenAI API key
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
2. Run the evaluation
```bash
python -m ragas_examples.evaluation_llm.evals
```

This will:

- Load the eligibility reasoning test dataset
- Run both baseline and candidate models on each test case
- Evaluate responses using accuracy metrics
- Generate detailed comparison results
- Save individual experiment results to CSV files

You can then inspect the results by opening the `experiments/` directory to see detailed per-case results for each model.
