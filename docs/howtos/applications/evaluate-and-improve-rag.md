# How to Evaluate and Improve a RAG App

In this guide, you'll learn how to evaluate and iteratively improve a RAG (Retrieval-Augmented Generation) app using Ragas.

## What you'll accomplish
- Set up evaluation dataset
- Establish metrics to measure RAG performance 
- Build a reusable evaluation pipeline
- Analyze errors and systematically improve your RAG app
- Learn how to leverage Ragas for RAG evaluation

## Test our RAG app

Install the dependencies:

```bash
uv pip install "ragas-examples[improverag]"
```

We've added tracing using MLflow to the RAG app. So you can see the traces in the MLflow UI.

Run the RAG app:

```bash
# Start mlflow server
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 
# Run the RAG app
export OPENAI_API_KEY="<your_key>"
uv run python -m ragas_examples.improve_rag.simple_rag
```
??? note "Output"
    ```bash
    uv run python -m ragas_examples.improve_rag.simple_rag
    ```
    ```bash

    Query: What architecture is the `tokenizers-linux-x64-musl` binary designed for?
    Loading dataset for BM25 retriever...
    Splitting documents for BM25 retriever...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2647/2647 [00:01<00:00, 1555.56it/s]
    Creating BM25 retriever...

    Answer: It’s built for the x86_64 architecture (specifically the x86_64-unknown-linux-musl target — 64-bit Linux with musl libc).

    Retrieved 3 documents:

    Document 1:
    Content: `tokenizers-linux-x64-musl`

    This is the **x86_64-unknown-linux-musl** binary for `tokenizers`...
    Source: tokenizers

    Document 2:
    Content: ## What are embeddings for?...
    Source: blog

    Document 3:
    Content: - What kind of model is it?
    - What is your model useful for?
    - What data was your model trained on?
    - How well does your model perform?...
    Source: transformers
    ```
You can view the traces on the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000).

![MLflow UI](../../_static/imgs/howto_improve_rag_mlflow.png)

## Create evaluation dataset

We'll be using [huggingface_doc_qa_eval](https://huggingface.co/datasets/m-ric/huggingface_doc_qa_eval) which is a dataset of questions and answers about the Hugging Face documentation. The evals script will download the dataset for you. 


## Set up metrics for RAG evaluation

It's better to start with simpler, focused metrics that directly measure your core use case. More information on metrics can be found in [Core Concepts - Metrics](../../concepts/metrics/index.md).

Here we use a `correctness` discrete metric that evaluates whether the RAG response contains the key information from the expected answer and is factually accurate based on the provided context.

```python
# examples/ragas_examples/improve_rag/evals.py
from ragas.metrics import DiscreteMetric

# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.
    
Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
    allowed_values=["pass", "fail"],
)
```

### The experiment function

The experiment function runs your RAG system on each data sample and evaluates the response using our correctness metric. More information on experimentation can be found in [Core Concepts - Experimentation](../../concepts/experimentation.md).

The experiment function takes a dataset row containing the question, expected context, and expected answer, then:

1. Queries the RAG system with the question
2. Evaluates the response using the correctness metric  
3. Returns detailed results including scores and reason

```python
# examples/ragas_examples/improve_rag/evals.py
import asyncio
from typing import Dict, Any
from ragas import experiment

@experiment()
async def evaluate_simple_rag(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run SimpleRAG evaluation on a single row.
    
    Args:
        row: Dictionary containing question, context, and expected_answer
        
    Returns:
        Dictionary with evaluation results
    """
    question = row["question"]
    
    # Get RAG response - using asyncio.to_thread for better concurrency
    rag_response = await asyncio.to_thread(rag_client.query, question, top_k=4)
    model_response = rag_response.get("answer", "")
    
    # Evaluate correctness - use asyncio.to_thread to avoid blocking the event loop
    score = await asyncio.to_thread(
        correctness_metric.score,
        question=question,
        expected_answer=row["expected_answer"],
        response=model_response,
        context=row["context"],
        llm=llm
    )
    
    # Return evaluation results
    result = {
        **row,
        "model_response": model_response,
        "correctness_score": score.value,
        "correctness_reason": score.reason,
        "num_retrieved_docs": rag_response.get("num_retrieved", 0),
        "retrieved_documents": [
            doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            for doc in rag_response.get("retrieved_documents", [])
        ]
    }
    
    return result
```

### Dataset preparation

The evaluation pipeline downloads the HuggingFace documentation Q&A dataset and converts it into a Ragas dataset format. Each row contains a question, the expected context, and the ground truth answer.

```python
# examples/ragas_examples/improve_rag/evals.py
import datasets
from pathlib import Path
from ragas import Dataset

def download_and_save_dataset() -> Path:
    """Download the HuggingFace doc Q&A dataset and save locally."""
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    dataset_path = datasets_dir / "hf_doc_qa_eval.csv"
    
    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    print("Downloading HuggingFace doc Q&A evaluation dataset...")
    hf_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
    
    # Convert to pandas and save as CSV
    df = hf_dataset.to_pandas()
    df.to_csv(dataset_path, index=False)
    
    return dataset_path

def create_ragas_dataset(dataset_path: Path) -> Dataset:
    """Create a Ragas Dataset from the downloaded CSV file."""
    dataset = Dataset(
        name="hf_doc_qa_eval",
        backend="local/csv",
        root_dir=".",
    )
    
    # Load the CSV data
    import pandas as pd
    df = pd.read_csv(dataset_path)
    
    # Add rows to the dataset
    for _, row in df.iterrows():
        dataset_row = {
            "question": row["question"],
            "context": row["context"],
            "expected_answer": row["expected_answer"],
        }
        dataset.append(dataset_row)
    
    # Save the dataset
    dataset.save()
    return dataset
```

## Run initial RAG experiment

Now let's run the complete evaluation pipeline to get baseline performance metrics for our RAG system:

```bash
# Install additional dependencies for evaluation
uv pip install "ragas-examples[improverag]"

# Run the evaluation (test mode for quicker results)
uv run python -m ragas_examples.improve_rag.evals --test
```

The `--test` flag runs evaluation on only the first 3 samples for quick testing. For full evaluation, run without the flag:

```bash
# Full evaluation (will take longer)
uv run python -m ragas_examples.improve_rag.evals
```

??? example "Full evaluation output"
    ```bash
    uv run python -m ragas_examples.improve_rag.evals
    === SimpleRAG Evaluation Pipeline ===

    1. Downloading dataset...
    Dataset already exists at datasets/hf_doc_qa_eval.csv

    2. Creating Ragas dataset...
    Created Ragas dataset with 65 samples

    2.5. Initializing BM25 retriever (this may take a moment)...
    Loading dataset for BM25 retriever...
    Splitting documents for BM25 retriever...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2647/2647 [00:00<00:00, 6365.97it/s]
    Creating BM25 retriever...
    BM25 retriever initialized successfully!

    3. Running evaluation on 65 samples...
    This may take several minutes depending on the dataset size...
    Running experiment: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [01:17<00:00,  1.19s/it]

    === Evaluation Results ===
    Total samples evaluated: 65
    Passed: 54
    Failed: 11
    Pass rate: 83.1%

    Evaluation completed successfully!
    ```

The evaluation results are saved to the `experiments/` directory with detailed information about each test case, including:

- **Model responses**: The actual answers generated by your RAG system
- **Correctness scores**: Pass/fail evaluation for each response
- **Correctness reasoning**: Detailed explanations for why responses passed or failed

## Analyze errors and failure modes

After running the evaluation, examine the results CSV file in the `experiments/` directory to identify patterns in failed cases. To view full details, you can go through the traces in MLflow for failed cases. 

### Analysis of actual failure patterns from our evaluation:

The core issue is **retrieval failure** - the BM25 retriever is not finding documents that contain the answers. The model correctly follows instructions to say when documents don't contain information, but the wrong documents are being retrieved.

**Poor Document Retrieval (Most Common Pattern)**
The BM25 retriever fails to retrieve relevant documents containing the answers:

| Question | Expected Answer | Model Response | Root Cause |
|----------|----------------|----------------|------------|
| "What is the default repository type for create_repo?" | `model` | "The provided documents do not state the default repository type..." | **BM25 missed docs with create_repo details** |
| "What is the purpose of the BLIP-Diffusion model?" | "controllable text-to-image generation and editing" | "The provided documents do not mention BLIP‑Diffusion..." | **BM25 didn't retrieve BLIP-Diffusion docs** |
| "What is the name of the new Hugging Face library for hosting scikit-learn models?" | `Skops` | "The provided documents do not mention or name any new Hugging Face library..." | **BM25 missed Skops documentation** |


## Improve your RAG system

While there are many improvements you can make in a `simple_rag` workflow, such as:

- Improving the chunking
- Improving the retrieval
- Adding vector embedding based retrieval

Sometimes, it's better to implement an Agentic solution. So we have created an `agentic_rag` version so that the agent can keep re-trying in multiple ways to find the right context from the data.

### Test the Agentic RAG Implementation

Run the agentic RAG app:

```bash
uv run python -m ragas_examples.improve_rag.agentic_rag
```

??? note "Output"
    ```bash
    uv run python -m ragas_examples.improve_rag.agentic_rag

    Question: What architecture is the `tokenizers-linux-x64-musl` binary designed for?

    ==================================================
    Loading dataset for BM25 retriever...
    Splitting documents for BM25 retriever...
    100%|██████████| 2647/2647 [00:00<00:00, 9152.51it/s]
    Creating BM25 retriever...
    Answer: The tokenizers-linux-x64-musl binary is designed for the x86_64 (64-bit Intel/AMD) architecture on Linux systems using the musl libc, which is common in lightweight distributions like Alpine Linux. (Source: tokenizers)
    ```

The key difference with the agentic approach is that the AI agent can:

- Call the BM25 retrieval tool multiple times with different search queries
- Iteratively refine its search strategy based on retrieved results  
- Decide when it has enough context to provide a comprehensive answer

Both implementations are instrumented with MLflow tracing, so you can view traces in the MLflow UI.

## Run experiment again and compare results

To evaluate the agentic RAG approach:

```bash
# Test mode with agentic RAG (recommended for quick testing)
uv run python -m ragas_examples.improve_rag.evals --agentic-rag --test

# Full evaluation with agentic RAG
uv run python -m ragas_examples.improve_rag.evals --agentic-rag
```

??? example "Agentic RAG evaluation output"
    ```bash
    uv run python -m ragas_examples.improve_rag.evals --agentic-rag
    🤖 Running with AGENTIC RAG
    === Agentic RAG Evaluation Pipeline ===

    1. Downloading dataset...
    Dataset already exists at datasets/hf_doc_qa_eval.csv

    2. Creating Ragas dataset...
    Created Ragas dataset with 65 samples

    2.5. Initializing BM25 retriever (this may take a moment)...
    Loading dataset for BM25 retriever...
    Splitting documents for BM25 retriever...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2647/2647 [00:00<00:00, 9151.95it/s]
    Creating BM25 retriever...
    BM25 retriever initialized successfully!

    3. Running Agentic RAG evaluation on 65 samples...
    This may take several minutes depending on the dataset size...
    Running experiment: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:49<00:00,  1.30it/s]

    === Evaluation Results ===
    Total samples evaluated: 65
    Passed: 61
    Failed: 4
    Pass rate: 93.8%

    Evaluation completed successfully!
    ```

### Performance Comparison

The agentic RAG approach shows significant improvement over the simple RAG baseline:

| Approach | Correctness | Improvement |
|----------|-----------|-------------|
| **Simple RAG** | 83.1% | - |
| **Agentic RAG** | **93.8%** | **+10.7%** |


## Apply this loop to your RAG system

Follow this systematic approach to improve any RAG system:

1. **Create evaluation dataset**: Use real queries from your system or generate synthetic data with LLMs. 

2. **Define metrics**: Choose simple metrics aligned with your use case (correctness, relevance, completeness). Keep it focused.

3. **Run baseline evaluation**: Measure current performance and analyze error patterns to identify systematic failures.

4. **Implement targeted improvements**: Based on error analysis, improve retrieval (chunking, hybrid search), generation (prompts, models), or try agentic approaches.

5. **Compare and iterate**: Test improvements against baseline. Change one thing at a time until accuracy meets business requirements.

The Ragas framework handles orchestration and result aggregation automatically, letting you focus on analysis and improvements rather than building evaluation infrastructure.
