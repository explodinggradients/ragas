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
from ragas.llms import llm_factory
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

# Define improved judge metric with enhanced evaluation criteria
accuracy_metric_v2 = DiscreteMetric(
    name="accuracy",
    prompt="""Evaluate if the response covers ALL the key concepts from the grading notes. Accept semantic equivalents but carefully check for missing concepts.

ABBREVIATION GUIDE - decode these correctly:
• Financial: val=valuation, post-$=post-money, rev=revenue, ARR/MRR=Annual/Monthly Recurring Revenue, COGS=Cost of Goods Sold, Opex=Operating Expenses, LTV=Lifetime Value, CAC=Customer Acquisition Cost
• Business: mkt=market, reg/regs=regulation/regulatory, corp gov=corporate governance, integr=integration, S&M=Sales & Marketing, R&D=Research & Development, acq=acquisition
• Technical: sys=system, elim=elimination, IP=Intellectual Property, TAM=Total Addressable Market, diff=differentiation
• Metrics: NPS=Net Promoter Score, SROI=Social Return on Investment, proj=projection, cert=certification

EVALUATION APPROACH:

Step 1 - Parse grading notes into distinct concepts:
- Separate by commas, semicolons, or line breaks
- Each item is a concept that must be verified
- Example: "*Gross Margin* >40%, CAC, LTV:CAC >3:1" = 3 concepts

Step 2 - For each concept, check if it's addressed:
- Accept semantic equivalents (e.g., "customer acquisition cost" = "CAC")
- Accept implicit coverage when it's clear (e.g., "revenue forecasting" covers "historical vs forecasted rev")
- Be flexible on exact numbers (e.g., "around 40%" acceptable for ">40%")

Step 3 - Count missing concepts:
- Missing 0 concepts = PASS
- Missing 1+ concepts = FAIL (even one genuinely missing concept should fail)
- Exception: If a long list (10+ items) has 1 very minor detail missing but all major points covered, use judgment

CRITICAL RULES:

1. Do NOT require exact wording - "market demand" = "mkt demand" = "demand analysis"

2. Markers (* or !) mean important, not mandatory exact phrases:
   - "*traction evidence*" can be satisfied by discussing metrics, growth, or validation
   - "!unbiased assumptions" can be satisfied by discussing assumption methodology

3. Numbers should be mentioned but accept approximations:
   - "$47B to $10B" can be "$47 billion dropped to around $10 billion"
   - "LTV:CAC >3:1" can be "LTV to CAC ratio of at least 3 to 1" or "3x or higher"

4. FAIL only when concepts are genuinely absent:
   - If notes mention "liquidation prefs, anti-dilution, board seats" but response only has board seats → FAIL
   - If notes mention "scalability, tech debt, IP" but response never discusses technical risks → FAIL
   - If notes mention "GDPR compliance" and response never mentions GDPR or EU regulations → FAIL

5. PASS when ALL concepts present:
   - All concepts covered, even with different wording → PASS
   - Concepts addressed implicitly when clearly implied → PASS
   - Minor phrasing differences → PASS
   - One or more concepts genuinely absent → FAIL

Response: {response}

Grading Notes: {grading_notes}

Are ALL distinct concepts from the grading notes covered in the response (accepting semantic equivalents and implicit coverage)?""",
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
    llm = llm_factory("gpt-4o-mini", client=openai_client)

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


async def main_v2():
    """Evaluate judge with improved v2 prompt."""
    # Load dataset
    dataset = load_dataset()
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Initialize LLM client
    logger.info("Initializing LLM client with model: gpt-4o-mini")
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o-mini", client=openai_client)

    # Run v2 evaluation with improved prompt
    logger.info("Running v2 evaluation with improved prompt...")
    results = await judge_experiment.arun(
        dataset,
        name="judge_accuracy_v2_gpt-4o-mini",
        accuracy_metric=accuracy_metric_v2,
        llm=llm,
    )
    
    passed = sum(1 for r in results if r["alignment"] == "pass")
    total = len(results)
    logger.info(f"✅ V2 alignment: {passed}/{total} passed ({passed/total:.1%})")
    
    return results


if __name__ == "__main__":
    import asyncio
    import sys
    
    # Run v2 if --v2 flag is passed, otherwise run baseline
    if len(sys.argv) > 1 and sys.argv[1] == "--v2":
        asyncio.run(main_v2())
    else:
        asyncio.run(main())
