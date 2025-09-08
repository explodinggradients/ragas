# How to evaluate a Text to SQL Agent

In this guide, you'll learn how to systematically evaluate and improve a text-to-SQL system using Ragas.

## What you'll accomplish
- Set up a baseline text-to-SQL system for evaluation
- Create evaluation metrics for SQL generation quality
- Build a reusable evaluation pipeline for your SQL agent  
- Implement improvements based on error analysis

## Setup your environment

Before evaluating your text-to-SQL agent, you need to set up the development environment with the required dependencies.

### Installation

We've created a simple module you can install and run so that you can focus on understanding the evaluation process instead of the setup.

```bash
uv pip install "ragas-examples[text2sql]"
```

!!! note "Full code"
    You can view the full code for the agent and evaluation pipeline [here](https://github.com/explodinggradients/ragas/tree/main/examples/ragas_examples/text2sql).

### Configuration

Set your API key:
```bash
export OPENAI_API_KEY=your-openai-api-key-here
```

For BookSQL dataset access (optional):
```bash
export HF_TOKEN=your-huggingface-token
```

### Verify your setup

Test that everything is working correctly:

```bash
# Test the text-to-SQL agent
python -m ragas_examples.text2sql.text2sql_agent --test
```

<details>
<summary>🔧 Expected test output</summary>

```
🧪 Running test with query: How much open credit does customer Andrew Bennett?
============================================================
Natural Query: How much open credit does customer Andrew Bennett?
Generated SQL: SELECT SUM(Open_balance) AS open_credit FROM master_txn_table WHERE Customers = 'Andrew Bennett';
Generation Time: 8608.11ms

============================================================
🔍 Executing SQL query against database...
❌ Error executing SQL: Failed to connect to database: Database file not found: BookSQL-files/BookSQL/accounting.sqlite

📝 Log exported to: text2sql_logs/run_test_20250908_142250_2025-09-08T14-22-50-469658.json
```

The SQL generation is working correctly, but execution fails because the database isn't set up yet. This is expected behavior - we'll set up the database in the next section.

</details>


## Prepare your dataset

### Create a dataset for your use case

Ideally you would want to continue this guide along with your own database and dataset. If you don't have your own database, you can use the BookSQL dataset we've taken as an example for this guide. 

If you have your own database, build a small, representative text-to-SQL dataset first:

1. Inventory your schema: list tables, columns, keys, and relationships. Prefer a stable snapshot (read-only).
2. Source questions: collect potential real user questions; add 10–20 synthetic paraphrases if needed. If you don't have real user questions, you can generate it using LLMs by sharing the business use case context.
3. Write ground-truth SQL: for each question, author correct SQL queries. Keep queries read-only (SELECT); validate they run and return results on your snapshot.
4. Ensure coverage ranging from simple to complex queries. It would help if you can the difficulty/priority of how important those queries are to your business use case. This will help us focus on the most important queries while evaluating our agent. 

### Download and examine the dataset

For this guide, we'll use the [BookSQL dataset](https://huggingface.co/datasets/Exploration-Lab/BookSQL), a comprehensive text-to-SQL evaluation dataset based on financial data. This dataset contains natural language questions paired with SQL queries for an accounting database. You are encouraged to use your own dataset for evaluation. If you already have a dataset, you can skip this section.

**Download the dataset:**

The text2sql module includes a convenient CLI tool to download the dataset:

```bash
# Download the dataset to ./BookSQL-files folder
python -m ragas_examples.text2sql.data_utils --download-data
```

**Note:** BookSQL is a gated dataset. If you encounter authentication errors:

1. Visit [the dataset page](https://huggingface.co/datasets/Exploration-Lab/BookSQL)
2. Accept the terms and conditions  
3. Run `huggingface-cli login` to authenticate
4. Try downloading again

**Examine the dataset structure:**

```bash
# Check the database schema
sqlite3 BookSQL-files/BookSQL/accounting.sqlite ".schema" | head -20
```

<details>
<summary>📋 Expected schema output</summary>

```sql
CREATE TABLE master_txn_table(
                    id INTEGER ,
                    businessID INTEGER NOT NULL ,
                    Transaction_ID INTEGER NOT NULL,
                    Transaction_DATE DATE NOT NULL,
                    Transaction_TYPE TEXT NOT NULL,
                    Amount DOUBLE NOT NULL,
                    CreatedDATE DATE NOT NULL,
                    CreatedUSER TEXT NOT NULL,
                    Account TEXT NOT NULL,
                    AR_paid TEXT,
                    AP_paid TEXT,
                    Due_DATE DATE,
                    Open_balance DOUBLE,
                    Customers TEXT,
                    Vendor TEXT,
                    Product_Service TEXT,
                    Quantity INTEGER,
                    Rate DOUBLE,
                    Credit DOUBLE,
```
</details>

The dataset contains:

- **Database**: SQLite file with accounting data (invoices, clients, etc.)
- **Questions**: Natural language queries in English
- **SQL**: Corresponding SQL queries
- **Difficulty levels**: Easy, Medium, Hard categories

### Create your evaluation subset

Once you have the BookSQL dataset downloaded, create a balanced evaluation subset to work with during development. 

**Create a balanced sample dataset:**

We've created a script to create a balanced evaluation subset. You can customize the number of samples per difficulty level.

```bash
# Create a CSV with custom number of samples per difficulty level
python -m ragas_examples.text2sql.data_utils --create-sample --samples 33 --validate --require-data
```

The `--validate` flag will validate the SQL queries before including them in the sample. The `--require-data` flag will only include queries that return actual data. We did this after realising that some of the queries in the BookSQL dataset don't return data making them not ideal for evaluation.

<details>
<summary>📋 Expected output</summary>

```
📖 Loading data from BookSQL-files/BookSQL/train.json...
📊 Loaded 70828 total records
🚂 Found 70828 train records
🔍 Removed 35189 duplicate records (same Query + SQL)
📊 35639 unique records remaining
📈 Difficulty distribution (after deduplication):
   • medium: 20576 records
   • hard: 11901 records
   • easy: 3162 records
✅ Added 33 validated 'easy' records
✅ Added 33 validated 'medium' records
✅ Added 33 validated 'hard' records
💾 Saved 99 records to datasets/booksql_sample.csv
📋 Final distribution:
   • medium: 33 records
   • hard: 33 records
   • easy: 33 records
```
</details>

This creates `datasets/booksql_sample.csv` with 99 total examples balanced across difficulty levels. The sampling process includes automatic deduplication to ensure no duplicate Query+SQL combinations are included in your evaluation dataset. 

**Examine your sample dataset:**

```bash
# View the first few rows to understand the structure
head -5 datasets/booksql_sample.csv
```


| Query                                                        | SQL                                                                                                                                                                                                                                    | Levels | split |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-------|
| What is the balance due from Richard Aguirre?                | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Richard Aguirre" )                                                                                               | medium | train |
| What is the balance due from Sarah Oconnor?                  | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Sarah Oconnor" )                                                                                                 | medium | train |
| What is my average invoice from Jeffrey Moore?               | select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Jeffrey Moore" and transaction_type = 'invoice')                                                                              | hard   | train |
| How much open credit does customer Andrew Bennett?           | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Andrew Bennett" )                                                                                                | easy   | train |

The CSV contains four columns:
- **Query**: Natural language question
- **SQL**: Ground truth SQL query
- **Levels**: Difficulty level (easy/medium/hard)
- **split**: Dataset split (all "train" for this sample)


## Set up your baseline text-to-SQL system

### Database utilities

We'll need a simple database interface to execute SQL queries and return results as pandas DataFrames for evaluation. The `db_utils.py` module handles this:

```python
# Basic usage - connects to BookSQL by default
from ragas_examples.text2sql.db_utils import SQLiteDB

with SQLiteDB() as db:
    success, result = db.execute_query("SELECT COUNT(*) FROM master_txn_table")
    if success:
        print(f"Query returned: {len(result)} rows")
```

The utility provides pandas DataFrame output for easy comparison, automatic SQL normalization (quotes, whitespace, date functions), and clear error messages for the evaluation loop.

**Test the database connection:**

```bash
uv run python -m ragas_examples.text2sql.db_utils --tables
```

<details>
<summary>📋 Expected output</summary>

```
=== Database Tables ===
  chart_of_accounts
  customers
  employees
  master_txn_table
  payment_method
  products
  vendors
```
</details>

### Create your Text-to-SQL agent

Now that we have database utilities set up, let's create a Text-to-SQL agent that converts natural language queries to SQL. Our agent will include comprehensive tracing and logging for evaluation purposes.

**Create the agent:**

The `text2sql_agent.py` module provides a complete OpenAI-powered agent with extensive logging capabilities designed for Ragas evaluation workflows.

```python
# File: examples/ragas_examples/text2sql/text2sql_agent.py
from ragas_examples.text2sql.text2sql_agent import Text2SQLAgent, get_default_agent

# Quick usage with defaults
agent = get_default_agent()
result = agent.generate_sql("How much open credit does customer John Smith have?")

print(f"Generated SQL: {result.generated_sql}")
print(f"Generation time: {result.generation_time_ms:.2f}ms")
```

**Key agent features:**

- **OpenAI Integration**: Uses GPT-4 for high-quality SQL generation
- **Schema-Aware**: Loads database schema from configurable prompt file  
- **Comprehensive Tracing**: Logs every step for evaluation and debugging
- **Batch Processing**: Handle multiple queries efficiently
- **Error Handling**: Graceful error recovery with detailed logging
- **Export Capabilities**: JSON logs compatible with Ragas evaluation

**Test the agent:**

```bash
# Single query test
export OPENAI_API_KEY="your-api-key-here"

uv run python -m ragas_examples.text2sql.text2sql_agent --test
```

<details>
<summary>📋 Expected output</summary>

```
🧪 Running test with query: How much open credit does customer Andrew Bennett?
============================================================
2025-08-29 01:08:04,703 - INFO - Generating SQL for query: How much open credit does customer Andrew Bennett? (Run ID: 20250829_010804_3971)
2025-08-29 01:08:07,014 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-08-29 01:08:07,022 - INFO - Successfully generated SQL (135 chars)
Natural Query: How much open credit does customer Andrew Bennett?
Generated SQL: select sum(open_balance) from (
select distinct transaction_id, open_balance
from master_txn_table
where customers = "Andrew Bennett"
)
Generation Time: 2318.54ms

============================================================
🔍 Executing SQL query against database...
✅ SQL execution successful!

📊 Query Results:
sum(open_balance)
             None

Rows returned: 1

📝 Log exported to: text2sql_logs/run_test_20250829_010807_2025-08-29T01-08-07-307323.json
```
</details>

The `--test` flag runs an end-to-end demonstration that generates SQL from a natural language query, executes it against the BookSQL database, and displays the results. This shows the complete pipeline from natural language to actual database output.

**Custom query example:**

```bash
# Run with custom query
uv run python -m ragas_examples.text2sql.text2sql_agent --query "How many invoices have we sent to Nathan Pineda?"
```

This generates SQL for your specific query and exports trace logs for evaluation.

### Create your prompt

Creating an effective prompt is crucial for text-to-SQL performance. Our approach focuses on providing complete database context in a concise, direct format that gives the language model everything it needs to generate accurate SQL queries.

**Step 1: Extract the database schema**

First, use the database utilities to get the complete schema information:

```bash
# Get complete schema with column details
uv run python -m ragas_examples.text2sql.db_utils --schema
```

<details>
<summary>📋 Expected schema output</summary>

```
=== Database Schema ===
       table_name          column_name data_type  not_null default_value  primary_key
chart_of_accounts                   id   INTEGER         0          None            1
chart_of_accounts           businessID   INTEGER         1          None            2
chart_of_accounts         Account_name      TEXT         1          None            3
chart_of_accounts         Account_type      TEXT         1          None            0
        customers                   id   INTEGER         0          None            1
        customers           businessID   INTEGER         1          None            2
        customers        customer_name      TEXT         1          None            3
... (continues for all 7 tables and 62 total columns)
```

</details>

This gives you the complete table structure that needs to be included in your prompt.

**Step 2: Gather business context**

Review the dataset documentation to understand the business domain:

```bash
# Read the BookSQL dataset documentation
cat BookSQL-files/BookSQL/README.md
```

This provides essential context about:
- What each table represents (transactions, customers, vendors, etc.)
- The business domain (accounting/financial data)
- How tables relate to each other

**Step 3: Design prompt structure**

Based on our requirements for a concise, direct prompt, we structure it as:

1. **Role definition**: Clear task description
2. **Database context**: Business domain explanation  
3. **Table purposes**: What each table contains
4. **Complete schema**: All tables with exact column details
5. **Simple instruction**: Convert query to SQL, return SQL only

**Step 4: Create the prompt file**

Create `prompt.txt` with the extracted information:

```bash
# Create the prompt file
touch prompt.txt
```

**Step 5: Write the prompt content**

Our prompt follows this template structure:

```text
You are a SQL query generator for a business accounting database. Convert natural language queries to SQL queries.

DATABASE CONTEXT:
This is an accounting database (accounting.sqlite) containing business transaction and entity data.

TABLES AND THEIR PURPOSE:
- master_txn_table: Main transaction records for all business transactions
- chart_of_accounts: Account names and their types for all businesses  
- products_service: Products/services and their types used by businesses
- customers: Customer records with billing/shipping details
- vendors: Vendor records with billing address details
- payment_method: Payment methods used by businesses
- employees: Employee details including name, ID, hire date

DATABASE SCHEMA:

[Complete schema with all tables and columns]

INSTRUCTIONS:
Convert the user's natural language query into a valid SQL SELECT query. Return only the SQL query, no explanations or formatting.
```

**Key design decisions:**

- **Schema inclusion**: We include the complete schema directly in the prompt rather than having the agent fetch it dynamically, ensuring consistent context
- **Business context**: We provide clear explanation of the accounting domain so the agent understands relationships between entities
- **Table purposes**: Each table gets a brief description of its role in the business model
- **Query patterns**: Include common business logic patterns (learned from initial failures)
- **Concise instruction**: Simple, direct instruction to convert natural language to SQL with no extra formatting
- **No examples**: We keep it simple without few-shot examples to start with a baseline approach

**Why Query Patterns Are Essential:**

Our initial prompt failed because schema alone isn't enough. When we tested the agent with "How much open credit does customer Andrew Bennett?", it generated:

```sql
SELECT Open_balance FROM customers WHERE customer_name = 'Andrew Bennett';
```

This failed with "no such column: Open_balance" because:
- The agent logically assumed customer info lives in the `customers` table
- But customer credit is actually calculated from transaction records in `master_txn_table`
- The `customers` table has `Balance` (not `Open_balance`) 

**The fix:** Add common query patterns to guide business logic:
```text
COMMON QUERY PATTERNS:
- Customer open credit: SELECT SUM(Open_balance) FROM master_txn_table WHERE Customers = "customer_name"  
- Customer transactions: SELECT * FROM master_txn_table WHERE Customers = "customer_name"
- Account balances: SELECT SUM(Credit - Debit) FROM master_txn_table WHERE Account = "account_name"
```

After adding these patterns, the same query correctly generated:
```sql
SELECT SUM(Open_balance) FROM master_txn_table WHERE Customers = "Andrew Bennett"
```

This demonstrates why text-to-SQL evaluation is inherently iterative - you discover domain-specific requirements through failure analysis and prompt refinement.

**Test your prompt:**

```bash
# Test the prompt with a sample query
uv run python -m ragas_examples.text2sql.text2sql_agent --test
```

<details>
<summary>📋 Expected test output</summary>

```
🧪 Running test with query: How much open credit does customer Andrew Bennett?
============================================================
Natural Query: How much open credit does customer Andrew Bennett?
Generated SQL: select sum(open_balance) from (
select distinct transaction_id, open_balance
from master_txn_table
where customers = "Andrew Bennett"
)
Generation Time: 2318.54ms
============================================================
🔍 Executing SQL query against database...
✅ SQL execution successful!
```

</details>

This confirms that your prompt successfully guides the agent to generate valid, executable SQL queries.

**Prompt customization:**

```bash
# Use custom prompt file
uv run python -m ragas_examples.text2sql.text2sql_agent --prompt custom_prompt.txt --query "Your query here"

# Or specify in code
agent = Text2SQLAgent(
    client=openai.OpenAI(),
    prompt_file="custom_prompt.txt"
)
```

The prompt creation process is designed to be systematic and repeatable for any database schema.

### Agent tracing and evaluation

The agent includes comprehensive tracing that captures every step of the SQL generation process. This is essential for evaluation and debugging.

**Trace events captured:**

1. **Initialization**: Agent startup and configuration
2. **Schema Loading**: Prompt file loading and validation
3. **LLM Calls**: Complete OpenAI API interactions
4. **SQL Generation**: Query cleaning and processing
5. **Errors**: Detailed error information with context

**Example trace export:**

```python
# Generate SQL with full tracing
agent = get_default_agent()
result = agent.generate_sql("How many invoices have we sent to John Doe?")

# Export traces
log_file = agent.export_traces_to_log("eval_run_001", "How many invoices...", result)
print(f"Traces saved to: {log_file}")
```

**Trace log format:**

```json
{
  "run_id": "eval_run_001",
  "timestamp": "2024-01-20T10:30:45.123456",
  "model_name": "gpt-4o", 
  "natural_query": "How many invoices have we sent to John Doe?",
  "generated_sql": "select count(distinct transaction_id) from master_txn_table where customers = \"John Doe\" and transaction_type = 'invoice'",
  "generation_time_ms": 1234.56,
  "traces": [
    {
      "event_type": "llm_call",
      "component": "openai_api",
      "data": {
        "model": "gpt-4o",
        "temperature": 0.0,
        "natural_query": "How many invoices have we sent to John Doe?"
      }
    },
    {
      "event_type": "llm_response", 
      "component": "openai_api",
      "data": {
        "generated_sql": "select count(distinct transaction_id)...",
        "usage": {
          "prompt_tokens": 892,
          "completion_tokens": 45,
          "total_tokens": 937
        }
      }
    }
  ]
}
```

These trace logs provide complete visibility into the agent's behavior and are designed to integrate seamlessly with Ragas evaluation pipelines.

## Define evaluation metrics

For text-to-SQL systems, we need metrics that evaluate both the technical correctness of generated SQL and the accuracy of results. We'll use two complementary metrics that provide clear insights into different types of failures.

### SQL validity and execution accuracy metrics

**SQL Validity Metric**: Tests whether the generated SQL can be parsed and executed without syntax errors. This catches basic SQL generation issues like invalid syntax, wrong column names, or malformed queries.

**Execution Accuracy Metric**: Compares the actual results between expected and predicted SQL queries using datacompy. This validates that both queries return the same data, which is the ultimate test of correctness.

The evaluation system uses a three-tier result classification:
- `"correct"`: Query succeeds and matches expected results  
- `"incorrect"`: Query succeeds but returns different results, or has syntax errors
- `"dataset_error"`: Ground truth query fails - indicates dataset quality issues that should be excluded from accuracy calculations

This classification is crucial because it separates real model failures from dataset quality problems, giving you accurate performance measurements.

### Setting up metric functions

Create your evaluation metrics using Ragas discrete metrics. These metrics will be called asynchronously during evaluation to enable parallel processing:

```python
# File: examples/ragas_examples/text2sql/evals.py
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult
from ragas_examples.text2sql.db_utils import execute_sql

@discrete_metric(name="sql_validity", allowed_values=["correct", "incorrect"])
def sql_validity(predicted_sql: str):
    """Check if the generated SQL is syntactically valid by attempting execution."""
    try:
        success, result = execute_sql(predicted_sql)
        if success:
            return MetricResult(
                value="correct",
                reason="SQL executed successfully without syntax errors"
            )
        else:
            return MetricResult(
                value="incorrect", 
                reason=f"SQL execution failed: {result}"
            )
    except Exception as e:
        return MetricResult(
            value="incorrect",
            reason=f"SQL validation failed with exception: {str(e)}"
        )

@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect", "dataset_error"])
def execution_accuracy(predicted_sql: str, expected_sql: str):
    """Compare execution results of predicted vs expected SQL using datacompy."""
    try:
        # Execute expected SQL first
        expected_success, expected_result = execute_sql(expected_sql)
        if not expected_success:
            return MetricResult(
                value="dataset_error",
                reason=f"Expected SQL failed to execute: {expected_result}"
            )
        
        # Execute predicted SQL
        predicted_success, predicted_result = execute_sql(predicted_sql)
        if not predicted_success:
            return MetricResult(
                value="incorrect",
                reason=f"Predicted SQL failed to execute: {predicted_result}"
            )
        
        # Both queries succeeded - compare DataFrames using datacompy
        if isinstance(expected_result, pd.DataFrame) and isinstance(predicted_result, pd.DataFrame):
            # Handle empty DataFrames
            if expected_result.empty and predicted_result.empty:
                return MetricResult(value="correct", reason="Both queries returned empty results")
            
            if expected_result.empty != predicted_result.empty:
                return MetricResult(
                    value="incorrect",
                    reason=f"Expected returned {len(expected_result)} rows, predicted returned {len(predicted_result)} rows"
                )
            
            # Use datacompy to compare DataFrames with index-based comparison
            comparison = datacompy.Compare(
                expected_result.reset_index(drop=True), 
                predicted_result.loc[:, expected_result.columns].copy().reset_index(drop=True),
                on_index=True,  # Compare row-by-row by index position
                abs_tol=1e-10,  # Very small tolerance for floating point comparison
                rel_tol=1e-10,
                df1_name='expected',
                df2_name='predicted'
            )
            
            if comparison.matches():
                return MetricResult(
                    value="correct",
                    reason=f"DataFrames match exactly ({len(expected_result)} rows, {len(expected_result.columns)} columns)"
                )
            else:
                return MetricResult(
                    value="incorrect",
                    reason="DataFrames do not match - different data returned"
                )
                
    except Exception as e:
        return MetricResult(
            value="dataset_error",
            reason=f"Execution accuracy evaluation failed: {str(e)}"
        )
```

**Key implementation details:**

- **datacompy integration**: Uses `on_index=True` for row-by-row comparison, perfect for SQL result validation
- **Column handling**: Automatically reorders predicted columns to match expected column order
- **Error classification**: Distinguishes between model errors (`incorrect`) and dataset issues (`dataset_error`)
- **Async-ready**: Metrics work seamlessly with `asyncio.to_thread` for parallel evaluation

### The experiment function

The experiment function orchestrates the complete evaluation pipeline - running the text-to-SQL agent and computing metrics for each query:

```python
# File: examples/ragas_examples/text2sql/evals.py
import asyncio
from ragas import experiment
from ragas_examples.text2sql.text2sql_agent import get_default_agent

@experiment()
async def text2sql_experiment(row, model: str, prompt_file: Optional[str], experiment_name: str):
    """Experiment function for text-to-SQL evaluation."""
    # Create text-to-SQL agent
    agent = get_default_agent(
        model_name=model,
        prompt_file=prompt_file,
        logdir="text2sql_logs"
    )
    
    # Generate SQL from natural language query (async to enable parallelism)
    result = await asyncio.to_thread(agent.generate_sql, row["Query"])
    
    # Score the response using our metrics (async to enable parallelism)  
    validity_score = await asyncio.to_thread(sql_validity.score, predicted_sql=result.generated_sql)
    accuracy_score = await asyncio.to_thread(
        execution_accuracy.score,
        predicted_sql=result.generated_sql,
        expected_sql=row["SQL"]
    )

    return {
        "id": row.get("id", f"row_{hash(row['Query']) % 10000}"),
        "query": row["Query"],
        "expected_sql": row["SQL"],
        "predicted_sql": result.generated_sql,
        "level": row["Levels"],
        "experiment_name": experiment_name,
        "sql_validity": validity_score.value,
        "execution_accuracy": accuracy_score.value,
        "validity_reason": validity_score.reason,
        "accuracy_reason": accuracy_score.reason,
    }
```

**Async implementation benefits:**
- **Parallel execution**: Multiple queries evaluated simultaneously 
- **`asyncio.to_thread`**: Wraps synchronous functions (agent, metrics) for async compatibility
- **Progress tracking**: Real-time progress bar shows evaluation speed

### Dataset loader

Load your evaluation dataset into a Ragas Dataset object for experiment execution:

```python
# File: examples/ragas_examples/text2sql/evals.py
from ragas import Dataset

def load_dataset(limit: Optional[int] = None):
    """Load the text-to-SQL dataset from CSV file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "datasets", "booksql_sample.csv")
    
    df = pd.read_csv(dataset_path)
    
    # Limit dataset size for testing (default 10, None means no limit)
    if limit is not None and limit > 0:
        df = df.head(limit)
    
    # Create Ragas Dataset
    dataset = Dataset(name="text2sql_booksql", backend="local/csv", root_dir=".")
    
    for _, row in df.iterrows():
        dataset.append({
            "Query": row["Query"],
            "SQL": row["SQL"], 
            "Levels": row["Levels"],
            "split": row["split"],
        })
    
    return dataset
```

The dataset loader includes a `limit` parameter for development workflows - start with small samples to catch basic errors quickly, then scale to full evaluation.

## Run baseline evaluation

### Execute evaluation pipeline and collect results

**Start with limited samples for faster iteration:**

For initial testing, run evaluations on a small subset to catch basic errors and validate your setup:

```bash
export OPENAI_API_KEY="your-api-key-here"

# Run evaluation on all samples (default behavior)
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini

# Test with limited sample size for quick validation
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini --limit 10
```

<details>
<summary>📋 Expected output (with --limit 10)</summary>

```
Loading dataset...
Dataset loaded with 10 samples (limited to 10 for testing)
Running text-to-SQL evaluation with model: gpt-5-mini
Running experiment: 100%|███████████████████████████████████████████████| 10/10 [00:16<00:00,  1.68s/it]
✅ text2sql_gpt_5_mini: 10 cases evaluated
Results saved to: /path/to/experiments/20250829-121726-text2sql_gpt_5_mini.csv
text2sql_gpt_5_mini SQL Validity: 100.00%
text2sql_gpt_5_mini Execution Accuracy: 20.00% (excluding 0 dataset errors)
```

</details>

**Run full evaluation once basics work:**

After validating your setup with limited samples, run the complete evaluation:

```bash
# Full dataset evaluation (all samples)  
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini --limit
```

**CLI options:**
- `--model`: OpenAI model to use (default: gpt-5-mini)
- `--prompt_file`: Custom prompt file (default: prompt.txt)
- `--limit`: Number of samples (default: all samples, specify a number to limit)
- `--name`: Custom experiment name for result tracking

### Initial performance analysis

The evaluation generates comprehensive CSV results for analysis. Each row contains:

**Core Data:**
- `query`: Natural language input
- `expected_sql`: Ground truth SQL  
- `predicted_sql`: Generated SQL
- `level`: Difficulty (easy/medium/hard)

**Metrics:**
- `sql_validity`: correct/incorrect (syntax and execution)  
- `execution_accuracy`: correct/incorrect/dataset_error (result comparison)

**Debugging Information:**
- `validity_reason`: Why SQL succeeded/failed
- `accuracy_reason`: Detailed comparison results

**Example baseline results:**

From our BookSQL evaluation with 10 samples:
- **SQL Validity: 100%** - All generated SQL queries were syntactically correct and executable
- **Execution Accuracy: 20%** - Only 2 out of 10 queries returned the same results as expected
- **Dataset Errors: 0%** - No issues with ground truth queries

**Common patterns observed:**
- **Column name mismatches**: Expected `sum(open_balance)` vs Generated `SUM(Open_balance)` or `balance_due`
- **Table selection**: Agent chose simplified queries vs expected complex subqueries
- **Business logic differences**: Different but valid approaches to the same question

**Next steps for improvement:**
1. Analyze specific failure cases in the CSV results
2. Identify patterns in query structure differences  
3. Refine prompts based on common error types
4. Test improvements with targeted examples

The baseline gives you a concrete starting point and identifies the main areas needing improvement. The 100% SQL validity shows your agent generates executable queries, while the 20% execution accuracy highlights opportunities for better result matching.

## Analyze errors and failure patterns

After running evaluations, you can analyze the failure patterns to understand where your Text2SQL agent is making mistakes. We provide an automated error analysis tool that uses AI to categorize errors, but **manual review is essential** to ensure you're improving in the right direction.

### Error Analysis Tool

The error analysis script automatically categorizes failures in your evaluation results:

```bash
uv run python -m ragas_examples.text2sql.analyze_errors --input experiments/your_results.csv
```

This will:

1. Process all rows where `execution_accuracy` is "incorrect"
2. Use OpenAI's GPT model to analyze each failure
3. Add two new columns: `error_analysis` (explanation) and `error_codes` (categorized errors)
4. Create an annotated CSV file with suffix `_annotated.csv`
5. Display a summary of error patterns

### Error Categories

The tool categorizes errors into these types:

| Error Code | Description |
|------------|-------------|
| `AGGR_DISTINCT_MISSING` | Used COUNT/SUM without DISTINCT or deduplication |
| `WRONG_FILTER_COLUMN` | Filtered on the wrong column |
| `WRONG_SOURCE_TABLE_OR_COLUMN` | Selected metric from the wrong table/column |
| `EXTRA_TRANSFORMATION_OR_CONDITION` | Added ABS(), extra filters that change results |
| `OUTPUT_COLUMN_ALIAS_MISMATCH` | Output column names don't match |
| `NULL_OR_EMPTY_RESULT` | Result is None/empty due to wrong filters or source |
| `GENERIC_VALUE_MISMATCH` | Aggregation computed but numeric value differs |
| `OTHER` | Fallback for unclassified errors |

### Example Output

```bash
Found 8 rows with incorrect execution accuracy
Processing batch 1/1 (8 rows)
  Starting analysis for row 0 (ID: row_1707)
  Starting analysis for row 2 (ID: row_3117)
  Starting analysis for row 3 (ID: row_9270)
  Starting analysis for row 4 (ID: row_6287)
  Starting analysis for row 6 (ID: row_9582)
  Starting analysis for row 7 (ID: row_7176)
  Starting analysis for row 8 (ID: row_3222)
  Starting analysis for row 9 (ID: row_5534)
  Waiting for 8 API calls to complete...
  ✓ Completed row 0: ['AGGR_DISTINCT_MISSING', 'OUTPUT_COLUMN_ALIAS_MISMATCH']
  ✓ Completed row 2: ['AGGR_DISTINCT_MISSING', 'OUTPUT_COLUMN_ALIAS_MISMATCH']
  ✓ Completed row 3: ['WRONG_SOURCE_TABLE_OR_COLUMN', 'WRONG_FILTER_COLUMN', 'OUTPUT_COLUMN_ALIAS_MISMATCH']
  ✓ Completed row 4: ['WRONG_SOURCE_TABLE_OR_COLUMN', 'WRONG_FILTER_COLUMN']
  ✓ Completed row 6: ['AGGR_DISTINCT_MISSING', 'WRONG_FILTER_COLUMN', 'NULL_OR_EMPTY_RESULT']
  ✓ Completed row 7: ['WRONG_SOURCE_TABLE_OR_COLUMN', 'WRONG_FILTER_COLUMN', 'OUTPUT_COLUMN_ALIAS_MISMATCH']
  ✓ Completed row 8: ['AGGR_DISTINCT_MISSING']
  ✓ Completed row 9: ['AGGR_DISTINCT_MISSING', 'EXTRA_TRANSFORMATION_OR_CONDITION', 'OUTPUT_COLUMN_ALIAS_MISMATCH', 'NULL_OR_EMPTY_RESULT']
Batch 1 completed successfully!
Analysis complete. Output written to: experiments/20250904-154438-text2sql_gpt_5_mini_annotated.csv

==================================================
ERROR CODE SUMMARY
==================================================
AGGR_DISTINCT_MISSING                 5
OUTPUT_COLUMN_ALIAS_MISMATCH          5
WRONG_FILTER_COLUMN                   4
WRONG_SOURCE_TABLE_OR_COLUMN          3
NULL_OR_EMPTY_RESULT                  2
EXTRA_TRANSFORMATION_OR_CONDITION     1
==================================================
```

### ⚠️ Critical: Manual Review Required

**The AI analysis is only a starting point.** You must manually review the categorized errors because:

- AI can misinterpret complex SQL logic differences
- Some errors might be incorrectly categorized
- The "correct" SQL might actually be wrong for your use case
- Context about your specific database schema matters

**Without manual review, you risk:**

- Fixing the wrong problems
- Making your agent worse by optimizing for incorrect patterns
- Missing the real root causes of failures

### Using the Error Analysis Tool in Your Workflow

The `analyze_errors.py` script is available as a standalone tool you can use throughout your development cycle:

```bash
# Analyze any evaluation results CSV
uv run python -m ragas_examples.text2sql.analyze_errors --input path/to/your/results.csv

# The tool will create an annotated version with error categorizations
# Always manually review the categorizations before making changes
```

**Remember**: Use this tool as a starting point for understanding failure patterns, but always validate the AI's categorizations through manual inspection before implementing fixes.

### Review Process

1. **Examine high-frequency error codes** from the summary
2. **Manually inspect 5-10 examples** of each major error type
3. **Verify the AI's categorization** matches your understanding
4. **Check if the "expected" SQL is actually correct** for your schema
5. **Look for patterns** the AI might have missed
6. **Prioritize fixes** based on impact and frequency

Only after manual validation should you use these insights to improve your prompts, few-shot examples, or agent architecture.

### Apply the analysis to your latest run

Follow this concise loop each time you iterate:

1. Annotate the most recent results file to get categorized errors:

```bash
uv run python -m ragas_examples.text2sql.analyze_errors --input experiments/<your_results>.csv
```

2. Open the generated `<your_results>_annotated.csv` and review:
- Error code summary (e.g., `AGGR_DISTINCT_MISSING`, `WRONG_FILTER_COLUMN`, `WRONG_SOURCE_TABLE_OR_COLUMN`, `OUTPUT_COLUMN_ALIAS_MISMATCH`)
- 3–5 representative rows per high-frequency code

3. Decide what to change in the prompt using generic rules, not per-row fixes. Avoid adding case-specific examples; prefer schema-grounded guardrails you can reuse across runs and databases.

Repeat this loop iteratively:
- Run → Annotate → Review → Decide generic guardrails → Update `prompt_vX.txt` → Re-run → Compare → Repeat.
- Keep guardrails concise and schema-grounded so improvements generalize without overfitting.
 - Version your prompts (`prompt_v2.txt`, `prompt_v3.txt`, `prompt_v4.txt`) and maintain a brief changelog per version.
 - Stop when execution accuracy plateaus across two consecutive iterations or meets your business threshold.

## Improve your system  

Use the findings from the analysis loop to strengthen your prompt with domain-generic, schema-grounded guidance—without overfitting to any single failure case.

### Create and use a new prompt version

We keep the baseline prompt intact and create a new version for iteration:

```bash
cp prompt.txt prompt_v2.txt
```

Edit `prompt_v2.txt` to include concise, reusable guardrails. Keep them generic enough to apply broadly while grounded in the provided schema:

- Use exact table and column names from the schema; do not invent fields
- Prefer transactional facts from `master_txn_table`; use entity tables for static attributes
- Map parties correctly in filters:
  - Customer-focused → filter on `Customers`
  - Vendor-focused → filter on `Vendor`
- Disambiguate events via `Transaction_TYPE` (e.g., invoices → `Transaction_TYPE = 'invoice'`)
- Avoid double-counting by deduplicating on `Transaction_ID` for counts and aggregates:
  - Counts: `count(distinct Transaction_ID)`
  - Aggregates: compute over a deduplicated subquery on `(Transaction_ID, metric_column)`
- For open credit/balance due per customer, aggregate `Open_balance` from `master_txn_table` filtered by `Customers` with deduplication
- Do not add extra transforms or filters (e.g., `abs()`, `< 0`) unless explicitly asked
- Keep a single `SELECT`; avoid aliases for final column names

We save this improved prompt as `prompt_v2.txt`.

### Re-run evaluation with the new prompt

```bash
export OPENAI_API_KEY="your-api-key-here"
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini --prompt_file prompt_v2.txt --limit 10
```

Review the new results CSV in `experiments/`. If accuracy improves, proceed to a larger run (remove `--limit`), then repeat the analyze → decide → improve loop. Keep guardrails generic and schema-grounded to avoid overfitting to specific rows.

### Continue iterating: Create prompt v3

Even with the major improvements in `prompt_v2.txt`, the 60% accuracy still leaves room for growth. A deeper analysis of the failures reveals several recurring patterns:

1.  **Misunderstanding of Financial Concepts**: The model consistently defaults to aggregating the `Amount` column instead of the correct `Credit` (for income) or `Debit` (for expenses) columns. It also often fails to `JOIN` with `chart_of_accounts` to filter by account type (e.g., 'Income').
2.  **Adding Unnecessary Transformations**: The model frequently complicates queries with unrequested `DISTINCT` clauses or extra filters (like `Transaction_TYPE = 'invoice'`), which alter the results.
3.  **Incorrect Column Selection**: For "show all transactions" queries, it often uses `SELECT *` instead of the expected `SELECT DISTINCT Transaction_ID`, leading to schema mismatches. It also generates the wrong column names for aggregations (e.g. `max(transaction_date)` instead of `transaction_date`).
4.  **Incomplete Filtering**: It often misses `OR` conditions (e.g., checking both `Customers` and `Vendor` for a transaction "with" someone) or filters on the wrong column entirely.

Based on this deeper analysis, create `prompt_v3.txt` with even more specific, schema-grounded guidelines to address these recurring issues:

```bash
cp prompt_v2.txt prompt_v3.txt
```

Key additions to `prompt_v3.txt`:

```text
### CORE QUERY GENERATION GUIDELINES

1.  **Use Correct Schema**: Use exact table and column names...
2.  **Simplicity First**: Keep the query as simple as possible...
...

### ADVANCED QUERY PATTERNS

5.  **Financial Queries (Revenue, Sales, Expenses)**:
    -   **Metric Selection**:
        -   For revenue, income, sales, or money **received**: aggregate the `Credit` column.
        -   For expenses, bills, or money **spent**: aggregate the `Debit` column.
        -   Use the `Amount` column only when...
    -   **Categorical Financial Queries**: For questions involving financial categories... you **MUST** `JOIN` `master_txn_table` with `chart_of_accounts`...

6.  **Filtering Logic**:
    -   **Ambiguous Parties**: For questions about transactions "with" or "involving" a person or company, you **MUST** check both `Customers` and `Vendor` columns. E.g., `WHERE Customers = 'Name' OR Vendor = 'Name'`.
    -   **Avoid Extra Filters**: Do not add implicit filters...

7.  **Column Selection and Naming**:
    -   **Avoid `SELECT *`**: When asked to "show all transactions", return only `DISTINCT Transaction_ID`...
    -   **"Most Recent" / "Last" Queries**: To get the 'most recent' or 'last' record, use `ORDER BY Transaction_DATE DESC LIMIT 1`. This preserves the original column names... Avoid using `MAX()`...

```

These new rules are designed to be generic but directly target the observed failure patterns.

**Re-run evaluation with `prompt_v3.txt`:**

```bash
export OPENAI_API_KEY="your-api-key-here"
uv run python -m ragas_examples.text2sql.evals run \
  --model gpt-5-mini \
  --prompt_file prompt_v3.txt \
  --name "gpt-5-mini-promptv3"
```

### Key principles for continued iteration

**The 70% accuracy achieved with `prompt_v3.txt` demonstrates the power of systematic iteration.** You can continue this process to push accuracy even higher:

1. **Analyze the remaining 30% of failures** using `analyze_errors.py` on your v3 results
2. **Identify new patterns** in the remaining incorrect queries  
3. **Create additional prompt versions** with targeted guidelines
4. **Re-evaluate and compare** to track incremental progress

**Key principles for continued iteration:**

- Each iteration should target **3-5 high-frequency error patterns** from the latest results
- Keep new rules **generic and schema-grounded** to avoid overfitting
- **Test with small samples first** (--limit 10) before full evaluation
- **Stop when accuracy plateaus** across 2-3 consecutive iterations
- **Document your changes** with a brief changelog per prompt version

This methodical approach can often push text-to-SQL accuracy well into the 80-90% range for well-scoped domains.

## Compare results

### Prompt v1 vs v2 (full dataset)

- After a quick 10-item smoke test, we observed higher execution accuracy with the improved prompt.
- We then ran the full dataset with both prompts:

| Prompt | SQL Validity | Execution Accuracy | Results CSV |
|---|---|---|---|
| v1 (`prompt.txt`) | 98.99% | 2.02% | `examples/ragas_examples/text2sql/experiments/20250905-151023-gpt-5-mini-promptv1.csv` |
| v2 (`prompt_v2.txt`) | 100.00% | 60.61% | `examples/ragas_examples/text2sql/experiments/20250905-150957-gpt-5-mini-promptv2.csv` |

These improvements came from generic, schema-grounded guardrails (not case-specific examples), so they should generalize without overfitting.

#### Decision criteria to pick a prompt

When comparing prompts (or models), choose the winner using this order:

1. Highest execution accuracy (primary)
2. Highest SQL validity (tiebreaker 1)
3. Fewest dataset errors (should be ~0) 
4. Lower average per-row latency and overall runtime (tiebreaker 2)
5. Stability across difficulty levels (no regressions on hard cases)

Keep a brief changelog for each prompt version (e.g., `prompt_v3.txt`, `prompt_v4.txt`) noting what guardrails changed.

#### Side-by-side CSV comparison

```bash
uv run python -m ragas_examples.text2sql.evals compare \
  --inputs experiments/<run1>.csv experiments/<run2>.csv \
  --output comparison.csv
```

This prints per-run metrics and writes a combined CSV with aligned rows for easy inspection.

### When to use an agentic retry loop

For older or weaker models, SQL validity may be low. In that case, add an agentic retry loop that:
- Returns DB error messages to the agent
- Asks the agent to correct the query and retry within a small budget

Since validity is already high in our runs, we are not enabling this flow here.

### Final Results Comparison

After running all prompt versions, we can compare the final results.

| Prompt | SQL Validity | Execution Accuracy | Results CSV |
|---|---|---|---|
| v1 (`prompt.txt`) | 98.99% | 2.02% | `...-promptv1.csv` |
| v2 (`prompt_v2.txt`) | 100.00% | 60.61% | `...-promptv2.csv` |
| v3 (`prompt_v3.txt`) | 98.99% | 70.71% | `20250905-164051-gpt-5-mini-promptv3.csv` |

**Progress Analysis:**
- **v1 → v2**: Massive 58 percentage point jump from 2.02% to 60.61% through basic deduplication and business logic guidelines
- **v2 → v3**: Additional 10 percentage point improvement from 60.61% to 70.71% through enhanced financial query guidelines, better filtering logic, and column selection rules
- The improvements target specific failure patterns identified through error analysis: financial concepts, unnecessary transformations, and incomplete filtering

These improvements came from generic, schema-grounded guardrails (not case-specific examples), so they should generalize without overfitting.


### Reproducible commands and outputs

Run on full dataset with prompt v1:

```bash
uv run python -m ragas_examples.text2sql.evals run \
  --model gpt-5-mini \
  --prompt_file prompt.txt \
  --name "gpt-5-mini-promptv1"
```

<details>
<summary>📋 Output (prompt v1)</summary>

```text
Please note that you are missing the optional dependency: fugue. If you need to
use this functionality it must be installed.
Please note that you are missing the optional dependency: snowflake. If you need to use this functionality it must be installed.
Please note that you are missing the optional dependency: spark. If you need to use this functionality it must be installed.
/Users/sanjeed/work/ragas-main/ragas/examples/ragas_examples/text2sql/evals.py:17: UserWarning: Python 3.12 and above currently is not supported by Spark and Ray. Please note that some functionality will not work and currently is not supported.
  import datacompy
Loading dataset...
Dataset loaded with 99 samples (full dataset)
Running text-to-SQL evaluation with model: gpt-5-mini
Using prompt file: prompt.txt
Running experiment: 100%|██████████████████████| 99/99 [01:06<00:00,  1.49it/s]
✅ gpt-5-mini-promptv1: 99 cases evaluated
Results saved to: /Users/sanjeed/work/ragas-main/ragas/examples/ragas_examples/text2sql/experiments/20250905-151023-gpt-5-mini-promptv1.csv
gpt-5-mini-promptv1 SQL Validity: 98.99%
gpt-5-mini-promptv1 Execution Accuracy: 2.02% (excluding 0 dataset errors)
/Users/sanjeed/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none/lib/python3.12/multiprocessing/resource_tracker.py:255: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d ')
```

</details>

Run on full dataset with prompt v2:

```bash
uv run python -m ragas_examples.text2sql.evals run \
  --model gpt-5-mini \
  --prompt_file prompt_v2.txt \
  --name "gpt-5-mini-promptv2"
```

<details>
<summary>📋 Output (prompt v2)</summary>

```text
warning: `VIRTUAL_ENV=.venv` does not match the project environment path `/Users/sanjeed/work/ragas-main/ragas/.venv` and will be ignored; use `--active` to target the active environment instead
Please note that you are missing the optional dependency: fugue. If you need to use this functionality it must be installed.
Please note that you are missing the optional dependency: snowflake. If you need to use this functionality it must be installed.
Please note that you are missing the optional dependency: spark. If you need to use this functionality it must be installed.
/Users/sanjeed/work/ragas-main/ragas/examples/ragas_examples/text2sql/evals.py:17: UserWarning: Python 3.12 and above currently is not supported by Spark and Ray. Please note that some functionality will not work and currently is not supported.
  import datacompy
Loading dataset...
Dataset loaded with 99 samples (full dataset)
Running text-to-SQL evaluation with model: gpt-5-mini
Using prompt file: prompt_v2.txt
Running experiment: 100%|██████████████████████| 99/99 [01:00<00:00,  1.63it/s]
✅ gpt-5-mini-promptv2: 99 cases evaluated
Results saved to: /Users/sanjeed/work/ragas-main/ragas/examples/ragas_examples/text2sql/experiments/20250905-150957-gpt-5-mini-promptv2.csv
gpt-5-mini-promptv2 SQL Validity: 100.00%
gpt-5-mini-promptv2 Execution Accuracy: 60.61% (excluding 0 dataset errors)
/Users/sanjeed/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none/lib/python3.12/multiprocessing/resource_tracker.py:255: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d ')
```

</details>

## Apply to your use case

Follow this checklist to adapt the workflow to your own database and dataset:

1. Prepare your database connection

```python
# examples/ragas_examples/text2sql/db_utils.py
# Use default BookSQL, or point to your DB path when initializing where applicable
from ragas_examples.text2sql.db_utils import SQLiteDB
db = SQLiteDB("/path/to/your/database.sqlite")
```

2. Extract your schema and build a prompt

```bash
uv run python -m ragas_examples.text2sql.db_utils --schema
# Copy the printed schema into your prompt file along with business context
cp prompt.txt prompt_v2.txt
```

3. Provide an evaluation dataset (CSV) with columns: `Query`, `SQL`, `Levels`, `split`

- If you already have this format, place it under `datasets/` and update `load_dataset()` if the filename differs.
- If not, create a CSV matching that schema or adapt the loader to your columns.

4. Run evaluation with your prompt and iterate

```bash
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini --prompt_file prompt_v2.txt --limit 10
uv run python -m ragas_examples.text2sql.analyze_errors --input experiments/<your_results>.csv
```

5. Improve generically, not per-row

- Add or refine guardrails in `prompt_v2.txt` based on high-frequency, validated errors.
- Avoid case-specific examples; keep rules reusable and grounded in your schema.

6. Scale up and compare

```bash
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini --prompt_file prompt_v2.txt
```

Use `compare` mode in `evals.py` (or your own spreadsheet) to compare runs side-by-side.

### Troubleshooting: Run stalls or hangs

If the evaluation seems to freeze near the end or a few rows take a very long time, enable timeouts and verbose logs to identify the slow stage:

```bash
uv run python -m ragas_examples.text2sql.evals run \
  --model gpt-5-mini \
  --verbose \
  --timeout-gen 60 \
  --timeout-sql 90 \
  --max-rows-compare 10000
```

- **What the flags do**:
  - **--timeout-gen**: max seconds for the LLM to generate SQL per row.
  - **--timeout-sql**: max seconds for SQL execution/validation and comparison per row.
  - **--max-rows-compare**: skip expensive `datacompy` comparisons when results are very large.
  - **--verbose**: prints per-row timings (gen/validity/accuracy) and lengths to spot outliers.

To debug a single problematic row quickly, run only that row by index:

```bash
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini --row 97 --verbose
```

### When Prompt Improvements Plateau

If you've iterated through several prompt versions and execution accuracy has plateaued (e.g., no improvement over 2-3 consecutive iterations), consider these next steps:

#### 1. Upgrade to a Better Model

Sometimes the base model capability is the bottleneck. Try:

```bash
# Test with more capable models
uv run python -m ragas_examples.text2sql.evals run --model gpt-4o --prompt_file prompt_v4.txt
uv run python -m ragas_examples.text2sql.evals run --model claude-3-5-sonnet --prompt_file prompt_v4.txt
```

Better models often understand complex business logic and database relationships more accurately, leading to significant accuracy improvements even with the same prompt.

#### 2. Advanced Prompt Optimization

For systematic prompt optimization, consider frameworks like [DSPy](https://github.com/stanfordnlp/dspy) that can automatically optimize prompts through:

- Few-shot example selection
- Chain-of-thought optimization  
- Multi-step reasoning patterns
- Automatic prompt tuning based on your evaluation metrics

**Note**: We'll cover DSPy integration and advanced optimization techniques in a separate guide.

#### 3. Hybrid Approaches

Combine multiple techniques:
- Better base model + optimized prompts
- Agentic retry loops with error feedback
- Multi-step reasoning (generate → validate → refine)
- Ensemble approaches with multiple models

## Conclusion

You can evolve text-to-SQL quality through a repeatable loop: analyze failures, introduce generic, schema-grounded guardrails, and re-evaluate. In our runs, moving from `prompt.txt` to `prompt_v2.txt` improved full-dataset execution accuracy from ~2% to ~60% while maintaining high SQL validity.

Adopt the same approach for your database by swapping in your schema and dataset, keeping guardrails concise and broadly applicable. If you encounter low SQL validity with older models, consider an agentic retry loop that uses DB error messages to guide automatic corrections; otherwise, stick to prompt-centric iteration.

When prompt improvements plateau, upgrading to more capable models or using advanced optimization frameworks like DSPy can unlock further gains. The `analyze_errors.py` tool helps identify patterns throughout this process, but always manually validate its categorizations before making changes.

Ragas makes this easy by handling datasets, metrics, and experiments end-to-end—parallel execution, clear per-row diagnostics, and side-by-side comparisons—so you can focus on improving your prompts and systems rather than building evaluation plumbing.