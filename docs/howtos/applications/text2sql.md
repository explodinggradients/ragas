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

## Prepare your dataset

### Ready-to-use sample dataset

We've prepared a balanced sample dataset with 99 examples (33 each of easy, medium, and hard queries) from the BookSQL dataset. You can start evaluating immediately or create your own dataset following the next section. 

**Examine the sample dataset:**

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

### Create a dataset for your use case

To create your own evaluation dataset:

1. **Schema inventory**: List tables, columns, keys, and relationships
2. **Question sourcing**: Collect real user questions or generate with LLMs
3. **Ground-truth SQL**: Write correct SELECT queries for each question
4. **Coverage**: Include simple to complex queries with difficulty ratings 

### Download and examine the dataset

For this guide, we'll use the [BookSQL dataset](https://huggingface.co/datasets/Exploration-Lab/BookSQL). Skip this section if you have your own dataset.

**Download the dataset:**

```bash
export HF_TOKEN=your-huggingface-token
uv run python -m ragas_examples.text2sql.data_utils --download-data
```

**Note:** BookSQL is gated. Visit [the dataset page](https://huggingface.co/datasets/Exploration-Lab/BookSQL), accept terms, and run `huggingface-cli login` if you encounter authentication errors.

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

Create a balanced evaluation subset:

```bash
uv run python -m ragas_examples.text2sql.data_utils --create-sample --samples 33 --validate --require-data
```

This creates a balanced CSV with validated queries that return actual data.

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

This creates `datasets/booksql_sample.csv` with 99 balanced examples across difficulty levels. 


## Set up your baseline text-to-SQL system

### Database utilities

The `db_utils.py` module provides a simple database interface:

```python
from ragas_examples.text2sql.db_utils import SQLiteDB

with SQLiteDB() as db:
    success, result = db.execute_query("SELECT COUNT(*) FROM master_txn_table")
    if success:
        print(f"Query returned: {len(result)} rows")
```

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

```python
from ragas_examples.text2sql.text2sql_agent import get_default_agent

agent = get_default_agent()
result = agent.generate_sql("How much open credit does customer John Smith have?")

print(f"Generated SQL: {result.generated_sql}")
print(f"Generation time: {result.generation_time_ms:.2f}ms")
```

**Test the agent:**

```bash
# Single query test
export OPENAI_API_KEY=your-openai-api-key-here

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

### Create your prompt

**Extract the database schema:**

```bash
uv run python -m ragas_examples.text2sql.db_utils --schema
```

<details>
<summary>📋 Expected schema output</summary>

```
=== Database Schema ===
             name  type                                     sql
chart_of_accounts table CREATE TABLE chart_of_accounts(
                         id INTEGER ,
                         businessID INTEGER NOT NULL,
                         Account_name TEXT NOT NULL,
                         Account_type TEXT NOT NULL,
                         PRIMARY KEY(id,businessID,Account_name)
                         )
        customers table CREATE TABLE customers(
                         id INTEGER ,
                         businessID INTEGER NOT NULL,
                         customer_name TEXT NOT NULL,
                         customer_full_name TEXT ,
                         ... (continues for all columns)
                         PRIMARY KEY(id,businessID,Customer_name)
                         )
... (continues for all 7 tables with complete DDL)
```

</details>

**Write the prompt content:**

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

DATABASE SCHEMA (DDL):

[Complete DDL statements for all tables]

INSTRUCTIONS:
Convert the user's natural language query into a valid SQL SELECT query. Return only the SQL query, no explanations or formatting.
```

Use `--prompt custom_prompt.txt` to test with different prompt versions.

## Define evaluation metrics

For text-to-SQL systems, we need metrics that evaluate the accuracy of results. We'll use execution accuracy as our primary metric to validate that generated SQL returns the correct data.

**Execution Accuracy Metric**: Compares the actual results between expected and predicted SQL queries using datacompy. This validates that both queries return the same data, which is the ultimate test of correctness.

The evaluation system classifies results as:
- `"correct"`: Query succeeds and matches expected results  
- `"incorrect"`: Query doesn't succeed or succeeds but returns wrong results


### Setting up metric functions

Create your evaluation metrics using [Ragas discrete metrics](/concepts/metrics/overview). 

```python
# File: examples/ragas_examples/text2sql/evals.py
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult
from ragas_examples.text2sql.db_utils import execute_sql

@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect"])
def execution_accuracy(expected_sql: str, predicted_success: bool, predicted_result):
    """Compare execution results of predicted vs expected SQL using datacompy."""
    try:
        # Execute expected SQL
        expected_success, expected_result = execute_sql(expected_sql)
        if not expected_success:
            return MetricResult(
                value="incorrect",
                reason=f"Expected SQL failed to execute: {expected_result}"
            )
        
        # If predicted SQL fails, it's incorrect
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
                predicted_result.reset_index(drop=True),
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
            value="incorrect",
            reason=f"Execution accuracy evaluation failed: {str(e)}"
        )
```

The execution accuracy metric uses datacompy for precise DataFrame comparison and classifies errors to distinguish model failures from dataset issues. 

### The experiment function

The [experiment function](/concepts/experimentation) orchestrates the complete evaluation pipeline - running the text-to-SQL agent and computing metrics for each query:

```python
# File: examples/ragas_examples/text2sql/evals.py
import asyncio
from ragas import experiment
from ragas_examples.text2sql.text2sql_agent import get_default_agent
from ragas_examples.text2sql.db_utils import execute_sql

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
    
    # Execute predicted SQL once to share results between metrics
    predicted_success, predicted_result = await asyncio.to_thread(execute_sql, result.generated_sql)
    
    # Score the response using execution accuracy
    accuracy_score = await asyncio.to_thread(
        execution_accuracy.score,
        expected_sql=row["SQL"],
        predicted_success=predicted_success,
        predicted_result=predicted_result
    )

    return {
        "id": row.get("id", f"row_{hash(row['Query']) % 10000}"),
        "query": row["Query"],
        "expected_sql": row["SQL"],
        "predicted_sql": result.generated_sql,
        "level": row["Levels"],
        "experiment_name": experiment_name,
        "execution_accuracy": accuracy_score.value,
        "accuracy_reason": accuracy_score.reason,
    }
```

The async implementation enables parallel query evaluation with real-time progress tracking.

### Dataset loader

Load your evaluation dataset into a [Ragas Dataset](/concepts/datasets) object for experiment execution:

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
Results saved to: experiments/20250829-121726-text2sql_gpt_5_mini.csv
text2sql_gpt_5_mini Execution Accuracy: 20.00%
```

</details>

**Run full evaluation once basics work:**

After validating your setup with limited samples, run the complete evaluation:

```bash
# Full dataset evaluation (all samples)  
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini
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
- `execution_accuracy`: correct/incorrect (result comparison)

**Debugging Information:**
- `validity_reason`: Why SQL succeeded/failed
- `accuracy_reason`: Detailed comparison results

**Example baseline results:**

From our BookSQL evaluation with 10 samples:
- **Execution Accuracy: 20%** - Only 2 out of 10 queries returned the same results as expected
- **Dataset Errors: 0%** - No issues with ground truth queries

**Common patterns observed:**
- **Column name mismatches**: Expected `sum(open_balance)` vs Generated `SUM(Open_balance)` or `balance_due`
- **Business logic differences**: Different but valid approaches to the same question

The baseline provides a concrete starting point with 20% execution accuracy, showing clear opportunities for improvement in result matching.

## Analyze errors and failure patterns

After running evaluations, you can analyze the failure patterns to understand where your Text2SQL agent is making mistakes. We provide an automated error analysis tool that uses LLMs to categorize errors, but **manual review is essential** to ensure you're improving in the right direction.

### Sample evaluation results

Here are some sample rows from our baseline evaluation results to give you a complete picture of what the evaluation data looks like:

| Query | Expected SQL | Predicted SQL | Level | Execution Accuracy |
|-------|--------------|---------------|-------|--------------|-------------------|
| What is the balance due from Richard Aguirre? | `select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Richard Aguirre" )` | `SELECT customer_name, customer_full_name, Balance FROM customers WHERE customer_name = 'Richard Aguirre' OR customer_full_name = 'Richard Aguirre';` | medium | incorrect |
| How much open credit does customer Natalie Myers? | `select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Natalie Myers" )` | `SELECT Balance FROM customers WHERE customer_name = 'Natalie Myers' OR customer_full_name = 'Natalie Myers';` | easy | incorrect |
| What is my average invoice from Ashley Thompson? | `select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Ashley Thompson" and transaction_type = 'invoice')` | `SELECT AVG(Amount) FROM master_txn_table WHERE Customers = 'Ashley Thompson' AND Transaction_TYPE = 'Invoice';` | hard | incorrect |
| Show number of transactions with Carol Smith | `select count(distinct transaction_id) from master_txn_table where customers = 'Carol Smith' or vendor = 'Carol Smith'` | `SELECT COUNT(*) AS num_transactions FROM master_txn_table WHERE Customers = 'Carol Smith';` | medium | incorrect |

**Key observations from these examples:**

- **Execution Accuracy: 0%** - None of the queries return the same results as expected
- **Common failure patterns:**
  - Using wrong tables (`customers` vs `master_txn_table`)
  - Missing deduplication (`count(distinct transaction_id)` vs `COUNT(*)`)
  - Incomplete filtering logic (missing `OR vendor = 'Name'` conditions)
  - Wrong column names (`Balance` vs `open_balance`)

This shows that while the agent generates valid SQL, it needs significant improvement in understanding the business logic and database schema relationships.

### Error Analysis

The error analysis script automatically categorizes failures in your evaluation results:

```bash
uv run python -m ragas_examples.text2sql.analyze_errors --input experiments/your_results.csv
```

This will:

1. Process all rows where `execution_accuracy` is "incorrect"
2. Use OpenAI's GPT 5 model to analyze each failure
3. Add two new columns: `error_analysis` (explanation) and `error_codes` (categorized errors)
4. Create an annotated CSV file with suffix `_annotated.csv`
5. Display a summary of error patterns

<details>
<summary>Example Output</summary>

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

</details>

### ⚠️ Critical: Manual Review Required

**The AI analysis is only a helper tool.** You must manually review the categorized errors because:

- AI can misinterpret complex SQL logic differences
- Some errors might be incorrectly categorized
- The "correct" SQL might actually be wrong for your use case
- Context about your specific database schema matters

**Without manual review, you risk:**

- Fixing the wrong problems
- Making your agent worse by optimizing for incorrect patterns
- Missing the real root causes of failures

If you feel like the AI is not doing a good job, you can manually review the errors and annoate them manually. Then you can just use an LLM to categorize them if required.

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

### Sample error analysis results (Prompt v2)

Here are some sample rows from the prompt v2 annotated results showing the types of errors that remain after the first round of improvements:

| Query | Expected SQL | Predicted SQL | Error Analysis | Error Codes |
|-------|--------------|---------------|----------------|-------------|
| How much we received from Fuel? | `select sum(credit) from master_txn_table as T1 join chart_of_accounts as T2 on T1.account = T2.account_name where account_type in ('Income', 'Other Income') and instr(account,"Fuel")` | `SELECT SUM(Amount) FROM (SELECT DISTINCT Transaction_ID, Amount FROM master_txn_table WHERE Customers = 'Fuel');` | The prediction summed Amount from master_txn_table instead of credit and omitted the join to chart_of_accounts, using Customers='Fuel' rather than filtering account names containing 'Fuel' and restricting to Income/Other Income. | `["WRONG_SOURCE_TABLE_OR_COLUMN", "WRONG_FILTER_COLUMN", "NULL_OR_EMPTY_RESULT"]` |
| What was the most recent invoice for Roberta Shaw? | `select transaction_id from master_txn_table where transaction_type = 'invoice' and customers = "Roberta Shaw" order by transaction_date desc limit 1` | `SELECT Transaction_ID, Transaction_DATE, Amount, Open_balance, CreatedDATE, CreatedUSER, Account, Customers, Product_Service FROM master_txn_table WHERE Transaction_TYPE = 'invoice' AND Customers = 'Roberta Shaw' ORDER BY Transaction_DATE DESC LIMIT 1;` | The prediction selected nine columns instead of only Transaction_ID as expected, causing a column/schema mismatch despite returning the same row. | `["OUTPUT_COLUMN_ALIAS_MISMATCH"]` |
| What is my average invoice from Jeremy Strong? | `select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Jeremy Strong" and transaction_type = 'invoice')` | `SELECT AVG(Amount) FROM (SELECT DISTINCT Transaction_ID, Amount FROM master_txn_table WHERE Transaction_TYPE = 'invoice' AND Vendor = 'Jeremy Strong') AS t;` | The query filters on Vendor = 'Jeremy Strong' instead of the correct customers column, so it likely matched no rows. This leads to AVG(amount) returning NULL. | `["WRONG_FILTER_COLUMN", "NULL_OR_EMPTY_RESULT"]` |

**Key observations from prompt v2 results:**

- **Significant improvement**: Execution accuracy jumped from 2% to 60% with basic deduplication and business logic guidelines
- **Remaining error patterns:**
  - **Missing OR conditions**: Queries about transactions "with" someone should check both `customers` and `vendor` columns
  - **Wrong column selection**: Using `Amount` instead of `credit` for financial queries
  - **Output schema mismatches**: Selecting too many columns or wrong column names
  - **Missing joins**: Not joining with `chart_of_accounts` for account-type filtering

These patterns inform the next iteration of prompt improvements, focusing on complete filtering logic and proper financial query handling.

1. Decide what to change in the prompt using generic rules, not per-row fixes. Avoid adding case-specific examples; prefer schema-grounded guardrails so that you are not overfitting to the data.

Repeat this loop iteratively:
- Run → Annotate → Review → Decide generic guardrails → Update `prompt_vX.txt` → Re-run → Compare → Repeat.
- Keep guardrails concise and schema-grounded so improvements generalize without overfitting.
 - Version your prompts (`prompt_v2.txt`, `prompt_v3.txt`, `prompt_v4.txt`) and maintain a brief changelog per version.
 - Stop when execution accuracy plateaus across two consecutive iterations or meets your business threshold.

## Improve your system  

### From error codes to prompt improvements

The error codes we created in our validation function directly inform our prompt improvements. Our analysis script categorizes failures into specific patterns, allowing us to address root causes systematically.

<details>
<summary>📋 Error taxonomy and prompt guardrails mapping</summary>

```python
# Error taxonomy → Root cause analysis → Prompt guardrails

# Aggregation issues (most common)
"AGGR_DISTINCT_MISSING" 
→ Model uses COUNT/SUM without deduplication
→ "Use count(distinct Transaction_ID) for counts and deduplicated subqueries for aggregates"

# Schema navigation errors  
"WRONG_FILTER_COLUMN" 
→ Model confuses customer vs vendor filtering logic
→ "Map parties correctly: Customer-focused → filter on Customers, Vendor-focused → filter on Vendor"

"WRONG_SOURCE_TABLE_OR_COLUMN"
→ Model invents fields or uses wrong tables
→ "Use exact table and column names from schema; prefer transactional facts from master_txn_table"

# Output format issues
"OUTPUT_COLUMN_ALIAS_MISMATCH"
→ Model adds unnecessary aliases that break result comparison
→ "Keep single SELECT; avoid aliases for final column names"

# Over-engineering
"EXTRA_TRANSFORMATION_OR_CONDITION" 
→ Model adds ABS(), extra filters that change intended results
→ "Do not add extra transforms unless explicitly asked"
```

</details>

This taxonomy-driven approach ensures our prompt improvements target actual failure modes with specific, actionable guardrails rather than generic advice.

### Create and use a new prompt version

We keep the baseline prompt intact and create a new version for iteration.

Create `prompt_v2.txt` to include concise, reusable guardrails. Keep them generic enough to apply broadly while grounded in the provided schema:

```text
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
```

We save this improved prompt as `prompt_v2.txt`.

### Re-run evaluation with the new prompt

```bash
export OPENAI_API_KEY="your-api-key-here"
uv run python -m ragas_examples.text2sql.evals run --model gpt-5-mini --prompt_file prompt_v2.txt --name "gpt-5-mini-promptv2"
```

Review the new results CSV in `experiments/` and continue the loop again.

### Continue iterating: Create prompt v3

Even with the major improvements in `prompt_v2.txt`, the 60% accuracy still leaves room for growth. A deeper analysis of the failures reveals several recurring patterns:

1.  **Misunderstanding of Financial Concepts**: The model consistently defaults to aggregating the `Amount` column instead of the correct `Credit` (for income) or `Debit` (for expenses) columns. It also often fails to `JOIN` with `chart_of_accounts` to filter by account type (e.g., 'Income').
2.  **Adding Unnecessary Transformations**: The model frequently complicates queries with unrequested `DISTINCT` clauses or extra filters (like `Transaction_TYPE = 'invoice'`), which alter the results.
3.  **Incorrect Column Selection**: For "show all transactions" queries, it often uses `SELECT *` instead of the expected `SELECT DISTINCT Transaction_ID`, leading to schema mismatches. It also generates the wrong column names for aggregations (e.g. `max(transaction_date)` instead of `transaction_date`).
4.  **Incomplete Filtering**: It often misses `OR` conditions (e.g., checking both `Customers` and `Vendor` for a transaction "with" someone) or filters on the wrong column entirely.

Based on this deeper analysis, create `prompt_v3.txt` with even more specific, schema-grounded guidelines to address these recurring issues:

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
- **Stop when accuracy plateaus** across 2-3 consecutive iterations
- **Document your changes** with a brief changelog per prompt version

## Compare results

### Prompt v1 vs v2 (full dataset)

- We ran the full dataset with both prompts:

| Prompt | Execution Accuracy | Results CSV |
|---|---|---|
| v1 (`prompt.txt`) | 2.02% | `experiments/20250905-151023-gpt-5-mini-promptv1.csv` |
| v2 (`prompt_v2.txt`) | 60.61% | `experiments/20250905-150957-gpt-5-mini-promptv2.csv` |

These improvements came from generic, schema-grounded guardrails (not case-specific examples), so they should generalize without overfitting.


### Final Results Comparison

After running all prompt versions, we can compare the final results.

| Prompt | Execution Accuracy | Results CSV |
|---|---|---|
| v1 (`prompt.txt`) | 2.02% | `experiments/...-promptv1.csv` |
| v2 (`prompt_v2.txt`) | 60.61% | `experiments/...-promptv2.csv` |
| v3 (`prompt_v3.txt`) | 70.71% | `experiments/...-promptv3.csv` |

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
Loading dataset...
Dataset loaded with 99 samples (full dataset)
Running text-to-SQL evaluation with model: gpt-5-mini
Using prompt file: prompt.txt
Running experiment: 100%|██████████████████████| 99/99 [01:06<00:00,  1.49it/s]
✅ gpt-5-mini-promptv1: 99 cases evaluated
Results saved to: experiments/20250905-151023-gpt-5-mini-promptv1.csv
gpt-5-mini-promptv1 Execution Accuracy: 2.02%
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
Loading dataset...
Dataset loaded with 99 samples (full dataset)
Running text-to-SQL evaluation with model: gpt-5-mini
Using prompt file: prompt_v2.txt
Running experiment: 100%|██████████████████████| 99/99 [01:00<00:00,  1.63it/s]
✅ gpt-5-mini-promptv2: 99 cases evaluated
Results saved to: experiments/20250905-150957-gpt-5-mini-promptv2.csv
gpt-5-mini-promptv2 Execution Accuracy: 60.61%
```

</details>


## Conclusion

This guide showed you how to build a systematic evaluation process for text-to-SQL agents using execution accuracy as the primary metric.

**Key takeaways:**

- Set up execution accuracy metrics to compare actual query results
- Follow the iterative process: evaluate → analyze → improve → repeat  
- Use manual error analysis to identify patterns and improve prompts
- Version your prompts and track improvements systematically

The evaluation framework gives you a reliable way to measure and improve your system, with Ragas handling the orchestration and result aggregation automatically.