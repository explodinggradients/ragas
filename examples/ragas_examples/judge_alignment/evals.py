"""
LLM-as-Judge alignment evaluation example.

Evaluates how well an LLM judge aligns with human judgments by:
- Using pre-existing responses from the dataset
- LLM judge evaluates each response
- Measuring alignment between judge and human labels
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas import Dataset, experiment
from ragas.llms import instructor_llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


# Define baseline judge metric with simple prompt
accuracy_metric = DiscreteMetric(
    name="accuracy",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\n\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)

# Define improved judge metric with few-shot examples
accuracy_metric_v2 = DiscreteMetric(
    name="accuracy",
    prompt="""TODO: Create this after error analysis""",
    allowed_values=["pass", "fail"],
)


def load_dataset(csv_path: Optional[Path] = None) -> Dataset:
    """Load annotated dataset with human judgments.
    
    Expected columns: question, grading_notes, response, target (pass/fail)
    """
    path = csv_path or (Path(__file__).resolve().parent / "datasets" / "benchmark_df.csv")
    df = pd.read_csv(path)

    dataset = Dataset(name="llm_judge_alignment", backend="local/csv", root_dir=".")
    
    for _, row in df.iterrows():
        dataset.append({
            "question": row["question"],
            "grading_notes": row["grading_notes"],
            "response": row["response"],
            "target": str(row["target"]).strip().lower(),
        })
    
    return dataset


@discrete_metric(name="judge_alignment", allowed_values=["pass", "fail"])
def judge_alignment(judge_label: str, human_label: str) -> MetricResult:
    """Compare judge decision with human label."""
    judge = judge_label.strip().lower()
    human = human_label.strip().lower()
    
    if judge == human:
        return MetricResult(value="pass", reason=f"Judge={judge}; Human={human}")
    
    return MetricResult(value="fail", reason=f"Judge={judge}; Human={human}")


@experiment()
async def judge_experiment(
    row: Dict[str, Any],
    accuracy_metric: DiscreteMetric,
    llm,
):
    """Run complete evaluation: Judge → Compare with human."""
    # Step 1: Get response (in production, this is where you'd call your LLM app)
    # For this evaluation, we use pre-existing responses from the dataset
    app_response = row["response"]
    
    # Step 2: Judge evaluates the response
    judge_score = await accuracy_metric.ascore(
        question=row["question"],
        grading_notes=row["grading_notes"],
        response=app_response,
        llm=llm,
    )

    # Step 3: Compare judge decision with human target
    alignment = judge_alignment.score(
        judge_label=judge_score.value,
        human_label=row["target"]
    )

    return {
        **row,
        "judge_label": judge_score.value,
        "judge_critique": judge_score.reason,
        "alignment": alignment.value,
        "alignment_reason": alignment.reason,
    }


async def main():
    """Example: evaluate judge with baseline prompt."""
    # Load dataset
    dataset = load_dataset()
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Initialize LLM client
    logger.info("Initializing LLM client with model: gpt-4o-mini")
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm = instructor_llm_factory("openai", model="gpt-4o-mini", client=openai_client)
    
    # Run baseline evaluation
    logger.info("Running baseline evaluation...")
    results = await judge_experiment.arun(
        dataset,
        name="judge_baseline_v1_gpt-4o-mini",
        accuracy_metric=accuracy_metric,
        llm=llm,
    )
    
    passed = sum(1 for r in results if r["alignment"] == "pass")
    total = len(results)
    logger.info(f"✅ Baseline alignment: {passed}/{total} passed ({passed/total:.1%})")
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
