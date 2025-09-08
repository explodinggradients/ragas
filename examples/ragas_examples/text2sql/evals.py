import argparse
import asyncio
import datetime
import os
import sys
import time
from typing import List, Optional

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
from .text2sql_agent import get_default_agent

dotenv.load_dotenv("../../../.env")

# Global guard for comparison size; configurable via CLI
MAX_ROWS_COMPARE = 10000


@discrete_metric(name="sql_validity", allowed_values=["correct", "incorrect"])
def sql_validity(predicted_success: bool, predicted_result):
    """Check if the generated SQL is syntactically valid based on execution results."""
    try:
        if predicted_success:
            return MetricResult(
                value="correct",
                reason="SQL executed successfully without syntax errors"
            )
        else:
            return MetricResult(
                value="incorrect",
                reason=f"SQL execution failed: {predicted_result}"
            )
    except Exception as e:
        return MetricResult(
            value="incorrect",
            reason=f"SQL validation failed with exception: {str(e)}"
        )


@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect", "dataset_error"])
def execution_accuracy(expected_sql: str, predicted_success: bool, predicted_result):
    """Compare execution results of predicted vs expected SQL using datacompy."""
    try:
        # Execute expected SQL
        expected_success, expected_result = execute_sql(expected_sql)
        
        # If expected SQL fails, this is a dataset error
        if not expected_success:
            return MetricResult(
                value="dataset_error",
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
                # If datacompy fails, report it as an evaluation error
                return MetricResult(
                    value="dataset_error",
                    reason=f"DataFrame comparison failed with datacompy: {str(comparison_error)}"
                )
        else:
            return MetricResult(
                value="incorrect",
                reason="One or both query results are not DataFrames"
            )
            
    except Exception as e:
        return MetricResult(
            value="dataset_error",
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
    agent = get_default_agent(
        model_name=model,
        prompt_file=prompt_file,
        logdir="text2sql_logs"
    )
    
    row_id = row.get("id", f"row_{hash(row['Query']) % 10000}")

    # Generate SQL from natural language query with timeout
    gen_start = time.perf_counter()
    class _Res:
        def __init__(self, sql: str):
            self.generated_sql = sql

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(agent.generate_sql, row["Query"]),
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

    # Score the response using our metrics with shared execution results
    val_start = time.perf_counter()
    try:
        validity_score = await asyncio.wait_for(
            asyncio.to_thread(sql_validity.score, predicted_success=predicted_success, predicted_result=predicted_result),
            timeout=timeout_sql,
        )
    except asyncio.TimeoutError:
        validity_score = MetricResult(value="incorrect", reason="SQL validity timed out")
    val_dur = time.perf_counter() - val_start

    acc_start = time.perf_counter()
    try:
        accuracy_score = await asyncio.wait_for(
            asyncio.to_thread(
                execution_accuracy.score,
                expected_sql=row["SQL"],
                predicted_success=predicted_success,
                predicted_result=predicted_result,
            ),
            timeout=timeout_sql,
        )
    except asyncio.TimeoutError:
        # Treat as dataset issue to avoid penalizing model when DB stalls
        accuracy_score = MetricResult(value="dataset_error", reason="Execution accuracy timed out")
    acc_dur = time.perf_counter() - acc_start

    if verbose:
        print(
            f"[row={row_id}] gen={gen_dur:.2f}s sql_exec={sql_exec_dur:.2f}s val={val_dur:.2f}s acc={acc_dur:.2f}s "
            f"query_len={len(row['Query'])} sql_len={len(result.generated_sql)}"
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
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Error: OpenAI API key not found!")
        print("Please set your API key: export OPENAI_API_KEY=your_actual_key")
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

    # Accuracy summary
    validity_accuracy = sum(1 for r in results if r["sql_validity"] == "correct") / max(1, len(results))
    
    # Execution accuracy (excluding dataset errors)
    execution_results = [r for r in results if r["execution_accuracy"] != "dataset_error"]
    execution_accuracy = sum(1 for r in execution_results if r["execution_accuracy"] == "correct") / max(1, len(execution_results))
    
    dataset_errors = sum(1 for r in results if r["execution_accuracy"] == "dataset_error")
    
    print(f"{exp_name} SQL Validity: {validity_accuracy:.2%}")
    print(f"{exp_name} Execution Accuracy: {execution_accuracy:.2%} (excluding {dataset_errors} dataset errors)")




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-to-SQL Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument(
        "--model", "-m", type=str, default="gpt-5-mini", help="Model to use for text-to-SQL generation"
    )
    run_parser.add_argument(
        "--prompt_file", "-p", type=str, default=None, help="Prompt file to use (defaults to prompt.txt)"
    )
    run_parser.add_argument(
        "--name", "-n", type=str, default=None, help="Experiment name (defaults to model name)"
    )
    run_parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Number of samples to evaluate (default: all samples, specify a number to limit)"
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose per-row timing logs"
    )
    run_parser.add_argument(
        "--timeout-gen", type=int, default=60, help="Timeout (seconds) for LLM generation"
    )
    run_parser.add_argument(
        "--timeout-sql", type=int, default=90, help="Timeout (seconds) for SQL validity/accuracy stages"
    )
    run_parser.add_argument(
        "--max-rows-compare", type=int, default=10000, help="Maximum rows allowed for datacompy comparison"
    )
    run_parser.add_argument(
        "--row", type=int, default=None, help="Run only a single dataset row by index (0-based)"
    )


    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        limit = args.limit if args.limit is not None and args.limit > 0 else None  # None, -1 or 0 means no limit
        asyncio.run(
            run_command(
                model=args.model,
                prompt_file=args.prompt_file,
                name=args.name,
                limit=limit,
                timeout_gen=getattr(args, "timeout_gen", 45),
                timeout_sql=getattr(args, "timeout_sql", 20),
                verbose=getattr(args, "verbose", False),
                max_rows_compare=getattr(args, "max_rows_compare", 10000),
                only_row=getattr(args, "row", None),
            )
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Force shutdown to avoid lingering non-daemon threads in test CLI context
        os._exit(0)
    else:
        parser.print_help()
        sys.exit(2)
