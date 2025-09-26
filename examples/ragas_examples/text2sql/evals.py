import asyncio
import datetime
import os
import sys
import time
from typing import Optional

import dotenv
import pandas as pd

from ragas import Dataset, experiment
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

try:
    import datacompy
except ImportError:
    raise ImportError("datacompy is required for execution accuracy. Install with: pip install datacompy")

from .db_utils import execute_sql
from .text2sql_agent import Text2SQLAgent

dotenv.load_dotenv("../../../.env")

# Global guard for comparison size; configurable via CLI
MAX_ROWS_COMPARE = 10000


def get_openai_client():
    """Get AsyncOpenAI client with proper error handling."""
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key: export OPENAI_API_KEY='your_key'"
        )
    return AsyncOpenAI(api_key=api_key)




@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect"])
def execution_accuracy(expected_sql: str, predicted_success: bool, predicted_result):
    """Compare execution results of predicted vs expected SQL using datacompy."""
    try:
        # Execute expected SQL
        expected_success, expected_result = execute_sql(expected_sql)
        
        # If expected SQL fails, it's incorrect
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
                return MetricResult(
                    value="correct",
                    reason="Both queries returned empty results"
                )
            
            # If one is empty and the other isn't, they're different
            if expected_result.empty != predicted_result.empty:
                return MetricResult(
                    value="incorrect",
                    reason=f"Expected returned {len(expected_result)} rows, predicted returned {len(predicted_result)} rows"
                )
            
            # Guard for very large results to avoid pathological comparisons
            try:
                if len(expected_result) > MAX_ROWS_COMPARE or len(predicted_result) > MAX_ROWS_COMPARE:
                    return MetricResult(
                        value="incorrect",
                        reason=(
                            f"Result too large to compare (expected_rows={len(expected_result)}, "
                            f"predicted_rows={len(predicted_result)}, max_rows={MAX_ROWS_COMPARE})"
                        ),
                    )
            except Exception:
                # If len() fails for any reason, fall back to comparison attempt
                pass

            # Use datacompy to compare DataFrames
            try:
                # Reset index to ensure clean comparison
                expected_clean = expected_result.reset_index(drop=True)
                predicted_clean = predicted_result.reset_index(drop=True)
                
                # Compare using datacompy with index-based comparison
                comparison = datacompy.Compare(
                    expected_clean, 
                    predicted_clean,
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
                        reason=f"DataFrames do not match. {comparison.report()}\nExpected: \n{expected_result}\nPredicted: \n{predicted_result}"
                    )
                    
            except Exception as comparison_error:
                # If datacompy fails, report it as incorrect
                return MetricResult(
                    value="incorrect",
                    reason=f"DataFrame comparison failed with datacompy: {str(comparison_error)}"
                )
        else:
            return MetricResult(
                value="incorrect",
                reason="One or both query results are not DataFrames"
            )
            
    except Exception as e:
        return MetricResult(
            value="incorrect",
            reason=f"Execution accuracy evaluation failed: {str(e)}"
        )


@experiment()
async def text2sql_experiment(
    row,
    model: str,
    prompt_file: Optional[str],
    experiment_name: str,
    timeout_gen: int = 60,
    timeout_sql: int = 90,
    verbose: bool = False,
):
    """Experiment function for text-to-SQL evaluation."""
    # Create text-to-SQL agent
    openai_client = get_openai_client()
    agent = Text2SQLAgent(
        client=openai_client,
        model_name=model,
        prompt_file=prompt_file
    )
    
    row_id = row.get("id", f"row_{hash(row['Query']) % 10000}")

    # Generate SQL from natural language query with timeout
    gen_start = time.perf_counter()
    class _Res:
        def __init__(self, sql: str):
            self.generated_sql = sql

    try:
        result = await asyncio.wait_for(
            agent.generate_sql(row["Query"]),
            timeout=timeout_gen,
        )
    except asyncio.TimeoutError:
        result = _Res("-- ERROR: generation timed out")
    gen_dur = time.perf_counter() - gen_start

    # Execute predicted SQL once to share results between metrics
    sql_exec_start = time.perf_counter()
    try:
        predicted_success, predicted_result = await asyncio.wait_for(
            asyncio.to_thread(execute_sql, result.generated_sql),
            timeout=timeout_sql,
        )
    except asyncio.TimeoutError:
        predicted_success, predicted_result = False, "SQL execution timed out"
    sql_exec_dur = time.perf_counter() - sql_exec_start

    # Score the response using execution accuracy
    acc_start = time.perf_counter()
    try:
        accuracy_score = await asyncio.wait_for(
            execution_accuracy.ascore(
                expected_sql=row["SQL"],
                predicted_success=predicted_success,
                predicted_result=predicted_result,
            ),
            timeout=timeout_sql,
        )
    except asyncio.TimeoutError:
        accuracy_score = MetricResult(value="incorrect", reason="Execution accuracy timed out")
    acc_dur = time.perf_counter() - acc_start

    if verbose:
        print(
            f"[row={row_id}] gen={gen_dur:.2f}s sql_exec={sql_exec_dur:.2f}s acc={acc_dur:.2f}s "
            f"query_len={len(row['Query'])} sql_len={len(result.generated_sql)}"
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


def load_dataset(limit: Optional[int] = None, only_row: Optional[int] = None):
    """Load the text-to-SQL dataset from CSV file."""
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "datasets", "booksql_sample.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Read CSV and create Dataset
    df = pd.read_csv(dataset_path)
    
    # If only a single row is requested
    if only_row is not None:
        if only_row < 0 or only_row >= len(df):
            raise IndexError(f"--row index out of range: {only_row} (dataset has {len(df)} rows)")
        df = df.iloc[[only_row]].copy()
    
    # Limit dataset size for testing (default 10, None means no limit)
    if limit is not None and limit > 0:
        df = df.head(limit)
    
    # Validate required columns
    required_cols = ["Query", "SQL", "Levels", "split"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
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




async def run_command(
    model: str,
    prompt_file: Optional[str],
    name: Optional[str],
    limit: Optional[int] = None,
    timeout_gen: int = 60,
    timeout_sql: int = 90,
    verbose: bool = False,
    max_rows_compare: int = 10000,
    only_row: Optional[int] = None,
) -> None:
    """Run a single experiment using the provided model and prompt file."""
    try:
        get_openai_client()  # Validate API key is available
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    print("Loading dataset...")
    dataset = load_dataset(limit=limit, only_row=only_row)
    if limit is not None:
        print(f"Dataset loaded with {len(dataset)} samples (limited to {limit} for testing)")
    else:
        print(f"Dataset loaded with {len(dataset)} samples (full dataset)")

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = name or f"text2sql_{model.replace('-', '_')}"

    # Ensure output directory exists (experiment framework saves under experiments/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(current_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    # Configure global comparison guard
    global MAX_ROWS_COMPARE
    MAX_ROWS_COMPARE = int(max_rows_compare)

    print(f"Running text-to-SQL evaluation with model: {model}")
    if prompt_file:
        print(f"Using prompt file: {prompt_file}")
    if verbose:
        print(
            f"Flags: timeout_gen={timeout_gen}s timeout_sql={timeout_sql}s max_rows_compare={MAX_ROWS_COMPARE}"
        )
        
    results = await text2sql_experiment.arun(
        dataset, 
        name=f"{run_id}-{exp_name}",
        model=model,
        prompt_file=prompt_file,
        experiment_name=exp_name,
        timeout_gen=timeout_gen,
        timeout_sql=timeout_sql,
        verbose=verbose,
    )
    
    print(f"✅ {exp_name}: {len(results)} cases evaluated")
    print(f"Results saved to: {os.path.join(experiments_dir, results.name)}.csv")

    # Execution accuracy summary
    execution_accuracy = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    
    print(f"{exp_name} Execution Accuracy: {execution_accuracy:.2%}")




# Demo
async def main():
    import os
    import pathlib
    from dotenv import load_dotenv
    
    # Load .env from root
    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    load_dotenv(root_dir / ".env")
    
    # Configure logging for demo
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Suppress HTTP request logs from OpenAI/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    logger.info("TEXT-TO-SQL EVALUATION DEMO")
    logger.info("=" * 40)
    
    # Run evaluation with limited samples for demo
    await run_command(
        model="gpt-5-mini",
        prompt_file=None,
        name="demo_evaluation",
        limit=5,  # Only evaluate 5 samples for demo
        timeout_gen=60,
        timeout_sql=90,
        verbose=False,
        max_rows_compare=10000,
        only_row=None,
    )


if __name__ == "__main__":
    asyncio.run(main())
