"""
AG-UI Agent Evaluation Script

This script demonstrates how to evaluate agents built with the AG-UI protocol
using Ragas metrics. It includes two evaluation scenarios:

1. Scientist Biographies - Tests factual correctness of agent responses
2. Weather Tool Usage - Tests tool calling accuracy

Prerequisites:
- An AG-UI compatible agent running at the specified endpoint URL
- See https://docs.ag-ui.com/quickstart/applications for agent setup

Usage:
    python evals.py --endpoint-url http://localhost:8000/agentic_chat
    python evals.py --endpoint-url http://localhost:8000/chat --skip-tool-eval
"""

import argparse
import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

from openai import AsyncOpenAI
from ragas.dataset_schema import (
    EvaluationDataset,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.integrations.ag_ui import evaluate_ag_ui_agent
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.messages import HumanMessage, ToolCall
from ragas.metrics import ToolCallF1
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    ResponseGroundedness,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
TEST_DATA_DIR = SCRIPT_DIR / "test_data"


def load_scientist_dataset() -> EvaluationDataset:
    """
    Load the scientist biographies dataset from CSV.

    Returns:
        EvaluationDataset with SingleTurnSample entries for testing factual correctness.
    """
    csv_path = TEST_DATA_DIR / "scientist_biographies.csv"
    logger.info(f"Loading scientist biographies dataset from {csv_path}")

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = SingleTurnSample(
                user_input=row["user_input"], reference=row["reference"]
            )
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} scientist biography samples")
    return EvaluationDataset(samples=samples)


def load_weather_dataset() -> EvaluationDataset:
    """
    Load the weather tool call dataset from CSV.

    Returns:
        EvaluationDataset with MultiTurnSample entries for testing tool call accuracy.
    """
    csv_path = TEST_DATA_DIR / "weather_tool_calls.csv"
    logger.info(f"Loading weather tool call dataset from {csv_path}")

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse the reference_tool_calls JSON
            tool_calls_data = json.loads(row["reference_tool_calls"])
            tool_calls = [
                ToolCall(name=tc["name"], args=tc["args"]) for tc in tool_calls_data
            ]

            # Create MultiTurnSample with user_input as a list of HumanMessage
            sample = MultiTurnSample(
                user_input=[HumanMessage(content=row["user_input"])],
                reference_tool_calls=tool_calls,
            )
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} weather tool call samples")
    return EvaluationDataset(samples=samples)


async def evaluate_scientist_biographies(
    endpoint_url: str, evaluator_llm: InstructorBaseRagasLLM
) -> tuple:
    """
    Evaluate the agent's ability to provide factually correct information
    about scientists.

    Args:
        endpoint_url: The AG-UI endpoint URL
        evaluator_llm: The LLM to use for evaluation

    Returns:
        Tuple of (result, dataframe) where result is the EvaluationResult
        and dataframe is the pandas DataFrame with results.
    """
    logger.info("=" * 80)
    logger.info("Starting Scientist Biographies Evaluation")
    logger.info("=" * 80)

    # Load dataset
    dataset = load_scientist_dataset()

    # Define metrics using the modern collections portfolio
    metrics = [
        FactualCorrectness(llm=evaluator_llm, mode="f1"),
        ContextPrecisionWithReference(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        ResponseGroundedness(llm=evaluator_llm),
    ]

    # Run evaluation
    logger.info(f"Evaluating against endpoint: {endpoint_url}")
    result = await evaluate_ag_ui_agent(
        endpoint_url=endpoint_url,
        dataset=dataset,
        metrics=metrics,
        evaluator_llm=evaluator_llm,
    )

    # Convert to DataFrame and clean up
    df = result.to_pandas()
    df = df.drop(columns=["retrieved_contexts"], errors="ignore")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Scientist Biographies Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"\n{df.to_string()}")

    metric_columns = [
        "factual_correctness(mode=f1)",
        "context_precision_with_reference",
        "context_recall",
        "response_groundedness",
    ]
    for column in metric_columns:
        if column in df.columns:
            logger.info(f"Average {column}: {df[column].mean():.4f}")

    if "factual_correctness(mode=f1)" in df.columns:
        logger.info(
            f"Perfect factual scores (1.0): {(df['factual_correctness(mode=f1)'] == 1.0).sum()}/{len(df)}"
        )

    return result, df


async def evaluate_weather_tool_use(
    endpoint_url: str, evaluator_llm: InstructorBaseRagasLLM
) -> tuple:
    """
    Evaluate the agent's ability to correctly call the weather tool.

    Args:
        endpoint_url: The AG-UI endpoint URL
        evaluator_llm: The LLM to use for evaluation

    Returns:
        Tuple of (result, dataframe) where result is the EvaluationResult
        and dataframe is the pandas DataFrame with results.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Starting Weather Tool Usage Evaluation")
    logger.info("=" * 80)

    # Load dataset
    dataset = load_weather_dataset()

    # Define metrics
    metrics = [ToolCallF1()]

    # Run evaluation
    logger.info(f"Evaluating against endpoint: {endpoint_url}")
    result = await evaluate_ag_ui_agent(
        endpoint_url=endpoint_url,
        dataset=dataset,
        metrics=metrics,
        evaluator_llm=evaluator_llm,
    )

    # Convert to DataFrame and clean up
    df = result.to_pandas()
    columns_to_drop = [
        col for col in ["retrieved_contexts", "reference"] if col in df.columns
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Weather Tool Usage Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"\n{df.to_string()}")

    if "tool_call_f1" in df.columns:
        avg_f1 = df["tool_call_f1"].mean()
        logger.info(f"\nAverage Tool Call F1: {avg_f1:.4f}")
        logger.info(
            f"Perfect scores (F1=1.0): {(df['tool_call_f1'] == 1.0).sum()}/{len(df)}"
        )
        logger.info(
            f"Failed scores (F1=0.0): {(df['tool_call_f1'] == 0.0).sum()}/{len(df)}"
        )

    return result, df


def save_results(df, scenario_name: str, output_dir: Path = None):
    """
    Save evaluation results to a timestamped CSV file.

    Args:
        df: The pandas DataFrame with evaluation results
        scenario_name: Name of the evaluation scenario
        output_dir: Directory to save results (defaults to script directory)
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scenario_name}_results_{timestamp}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    logger.info(f"\nResults saved to: {filepath}")


async def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate AG-UI agents using Ragas metrics"
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="http://localhost:8000/agentic_chat",
        help="AG-UI endpoint URL (default: http://localhost:8000/agentic_chat)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip-factual",
        action="store_true",
        help="Skip the factual correctness evaluation",
    )
    parser.add_argument(
        "--skip-tool-eval",
        action="store_true",
        help="Skip the tool call evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: script directory)",
    )

    args = parser.parse_args()

    # Setup evaluator LLM
    logger.info(f"Setting up evaluator LLM: {args.evaluator_model}")
    client = AsyncOpenAI()
    evaluator_llm = llm_factory(args.evaluator_model, client=client)

    # Run evaluations
    try:
        if not args.skip_factual:
            result, df = await evaluate_scientist_biographies(
                args.endpoint_url, evaluator_llm
            )
            save_results(df, "scientist_biographies", args.output_dir)

        if not args.skip_tool_eval:
            result, df = await evaluate_weather_tool_use(
                args.endpoint_url, evaluator_llm
            )
            save_results(df, "weather_tool_calls", args.output_dir)

        logger.info("\n" + "=" * 80)
        logger.info("All evaluations completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nEvaluation failed with error: {e}")
        logger.error(
            "\nPlease ensure your AG-UI agent is running at the specified endpoint."
        )
        logger.error(
            "See https://docs.ag-ui.com/quickstart/applications for setup instructions."
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
