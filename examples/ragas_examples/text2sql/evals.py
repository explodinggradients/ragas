import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas import Dataset, experiment
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

import datacompy

from .db_utils import execute_sql
from .text2sql_agent import Text2SQLAgent

# Load environment variables
load_dotenv(".env")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

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
            if len(expected_result) > 10000 or len(predicted_result) > 10000:
                return MetricResult(
                    value="incorrect",
                    reason=(
                        f"Result too large to compare (expected_rows={len(expected_result)}, "
                        f"predicted_rows={len(predicted_result)}, max_rows=10000)"
                    ),
                )

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
):
    """Experiment function for text-to-SQL evaluation."""
    # Create text-to-SQL agent
    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    agent = Text2SQLAgent(
        client=openai_client,
        model_name=model,
        prompt_file=prompt_file
    )
    
    # Generate SQL from natural language query
    result = await agent.query(row["Query"])

    # Execute predicted SQL
    try:
        predicted_success, predicted_result = execute_sql(result["sql"])
    except Exception as e:
        predicted_success, predicted_result = False, f"SQL execution failed: {str(e)}"

    # Score the response using execution accuracy
    accuracy_score = await execution_accuracy.ascore(
        expected_sql=row["SQL"],
        predicted_success=predicted_success,
        predicted_result=predicted_result,
    )

    return {
        "query": row["Query"],
        "expected_sql": row["SQL"],
        "predicted_sql": result["sql"],
        "level": row["Levels"],
        "execution_accuracy": accuracy_score.value,
        "accuracy_reason": accuracy_score.reason,
    }


def load_dataset(limit: Optional[int] = None):
    """Load the text-to-SQL dataset from CSV file."""
    dataset_path = Path(__file__).parent / "datasets" / "booksql_sample.csv"
    
    # Read CSV
    df = pd.read_csv(dataset_path)
    
    # Limit dataset size if requested
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


async def main():
    """Simple demo script to run text-to-SQL evaluation."""
    logger.info("TEXT-TO-SQL EVALUATION DEMO")
    logger.info("=" * 40)
    
    # Configuration
    model = "gpt-5-mini"
    prompt_file = None
    name = "demo_evaluation"
    limit = 5  # Only evaluate 5 samples for demo
    
    # Validate API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("❌ Error: OPENAI_API_KEY environment variable is not set")
        return

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(limit=limit)
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    logger.info(f"Running text-to-SQL evaluation with model: {model}")
        
    # Run the experiment
    results = await text2sql_experiment.arun(
        dataset, 
        name=name,
        model=model,
        prompt_file=prompt_file,
    )
    
    # Report results
    logger.info(f"✅ {name}: {len(results)} cases evaluated")

    # Calculate and display accuracy
    accuracy_rate = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    logger.info(f"{name} Execution Accuracy: {accuracy_rate:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
