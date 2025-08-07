# Datasets and Experiment Results

When we evaluate AI systems, we typically work with two main types of data:

1. **Evaluation Datasets**: These are stored under the `datasets` directory.
2. **Evaluation Results**: These are stored under the `experiments` directory.

## Evaluation Datasets

A dataset for evaluations contains:

1. Inputs: a set of inputs that the system will process.
2. Expected outputs (Optional): the expected outputs or responses from the system for the given inputs.
3. Metadata (Optional): additional information that can be stored alongside the dataset.

For example, in a Retrieval-Augmented Generation (RAG) system it might include query (input to the system), Grading notes (to grade the output from the system), and metadata like query complexity.

Metadata is particularly useful for slicing and dicing the dataset, allowing you to analyze results across different facets. For instance, you might want to see how your system performs on complex queries versus simple ones, or how it handles different languages.

## Experiment Results

Experiment results include:

1. All attributes from the dataset.
2. The response from the evaluated system.
3. Results of metrics.
4. Optional metadata, such as a URI pointing to the system trace for a given input.

For example, in a RAG system, the results might include Query, Grading notes, Response, Accuracy score (metric), link to the system trace, etc.

## Data Storage in Ragas

We understand that different teams have diverse preferences for organizing, updating, and maintaining data, for example:

- A single developer might store datasets as CSV files in the local filesystem.
- A small-to-medium team might use Google Sheets or Notion databases.
- Enterprise teams might rely on Box or Microsoft OneDrive, depending on their data storage and sharing policies.

Teams may also use various file formats like CSV, XLSX, or JSON. Among these, CSV or spreadsheet formats are often preferred for evaluation datasets due to their simplicity and smaller size compared to training datasets.

Ragas, as an evaluation framework, supports these diverse preferences by enabling you to use your preferred file systems and formats for storing and reading datasets and experiment results.

To achieve this, Ragas introduces the concept of **plug-and-play backends** for data storage:

- Ragas provides default backends like `local/csv` and `google_drive/csv`.
- These backends are extensible, allowing you to implement custom backends for any file system or format (e.g., `box/csv`).


## Using Datasets and Results via API

### Loading a Dataset

```python
from ragas_experimental import Dataset

test_dataset = Dataset.load(name="test_dataset", backend="local/csv", root_dir=".")
```

This command loads a dataset named `test_dataset.csv` from the `root_directory/datasets` directory. The backend can be any backend registered via Ragas backends.

### Loading Experiment Results

```python
from ragas_experimental import Experiment

experiment_results = Experiment.load(name="first_experiment", backend="local/csv", root_dir=".")
```

This command loads experiment results named `first_experiment.csv` from the `root_directory/experiments` directory. The backend can be any backend registered via Ragas backends.

## Data Validation Using Pydantic

Ragas provides data type validation via Pydantic. You can configure a preferred `data_model` for a dataset or experiment results to ensure data is validated before reading or writing to the data storage.

**Example**:

```python
from ragas_experimental import Dataset
from pydantic import BaseModel

class MyDataset(BaseModel):
    query: str
    ground_truth: str

test_dataset = Dataset.load(name="test_dataset", backend="local/csv", root_dir=".", data_model=MyDataset)
```

This ensures that the data meets the specified type requirements, preventing invalid data from being read or written.