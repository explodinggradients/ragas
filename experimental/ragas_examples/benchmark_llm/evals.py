import json
from ragas_experimental import Dataset, experiment
from ragas_experimental.metrics.result import MetricResult
from ragas_experimental.metrics.discrete import discrete_metric

from .prompt import run_prompt
from .config import BASELINE_MODEL, CANDIDATE_MODEL


def parse_eligibility_response(response: str):
    """Parse the JSON model response to extract eligibility, discount, and reason."""
    try:
        # Try to extract JSON from response (in case there's extra text)
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            return {
                'eligible': parsed.get('eligible'),
                'discount': parsed.get('discount_percentage'),
                'reason': parsed.get('reason', ''),
                'applied_rules': parsed.get('applied_rules', [])
            }
        else:
            # Fallback if no JSON found
            return {
                'eligible': None,
                'discount': None,
                'reason': response,
                'applied_rules': []
            }
    except (json.JSONDecodeError, Exception):
        return {
            'eligible': None,
            'discount': None,
            'reason': response,
            'applied_rules': []
        }


@discrete_metric(name="eligibility_accuracy", allowed_values=["correct", "incorrect"])
def eligibility_accuracy(prediction: str, expected_eligible, expected_discount):
    """Check if the eligibility and discount prediction is correct."""
    parsed = parse_eligibility_response(prediction)
    
    # Convert expected values to correct types (CSV loads as strings)
    expected_eligible_bool = str(expected_eligible).lower() == 'true'
    expected_discount_int = int(expected_discount)
    
    eligible_correct = parsed['eligible'] == expected_eligible_bool
    discount_correct = parsed['discount'] == expected_discount_int
    
    if eligible_correct and discount_correct:
        return MetricResult(
            value="correct", 
            reason=f"Correctly identified eligible={expected_eligible_bool}, discount={expected_discount_int}%"
        )
    else:
        return MetricResult(
            value="incorrect",
            reason=f"Expected eligible={expected_eligible_bool}, discount={expected_discount_int}%; Got eligible={parsed['eligible']}, discount={parsed['discount']}%"
        )


def create_benchmark_experiment(model_name: str):
    """Factory function to create experiment functions for different models."""
    
    @experiment()
    async def benchmark_experiment(row):
        # Get model response
        response = run_prompt(row["customer_profile"], model=model_name)
        
        # Parse response
        parsed = parse_eligibility_response(response)
        
        # Score the response
        score = eligibility_accuracy.score(
            prediction=response,
            expected_eligible=row["expected_eligible"],
            expected_discount=row["expected_discount"]
        )
        
        return {
            **row,
            "model": model_name,
            "response": response,
            "predicted_eligible": parsed['eligible'],
            "predicted_discount": parsed['discount'],
            "reasoning": parsed['reason'],
            "applied_rules": parsed['applied_rules'],
            "score": score.value,
            "score_reason": score.reason
        }
    
    return benchmark_experiment


def load_dataset():
    """Load the dataset from CSV file."""
    import os
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset = Dataset.load(
        name="eligibility_benchmark",
        backend="local/csv",
        root_dir=current_dir
    )
    return dataset


async def main():
    """Run the full benchmark comparing baseline vs candidate model."""
    # Check API key first
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Error: OpenAI API key not found!")
        print("Please set your API key: export OPENAI_API_KEY=your_actual_key")
        return
    
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create experiment functions for each model
    baseline_experiment = create_benchmark_experiment(BASELINE_MODEL)
    candidate_experiment = create_benchmark_experiment(CANDIDATE_MODEL)
    
    print(f"Running baseline model evaluation ({BASELINE_MODEL})...")
    baseline_results = await baseline_experiment.arun(dataset, name=f"{BASELINE_MODEL}")
    print(f"✅ {BASELINE_MODEL}: {len(baseline_results)} cases evaluated")
    print(f"   Results saved to: experiments/{baseline_results.name}.csv")
    
    print(f"\nRunning candidate model evaluation ({CANDIDATE_MODEL})...")
    candidate_results = await candidate_experiment.arun(dataset, name=f"{CANDIDATE_MODEL}")
    print(f"✅ {CANDIDATE_MODEL}: {len(candidate_results)} cases evaluated")
    print(f"   Results saved to: experiments/{candidate_results.name}.csv")

    # Check if we have results
    if len(baseline_results) == 0 or len(candidate_results) == 0:
        print("❌ No results returned. Please check:")
        print("1. OpenAI API key is set: export OPENAI_API_KEY=your_key")
        print("2. Dataset loaded correctly")
        print("3. Network connectivity")
        return
    
    # Calculate accuracy scores
    baseline_accuracy = sum(1 for r in baseline_results if r["score"] == "correct") / len(baseline_results)
    candidate_accuracy = sum(1 for r in candidate_results if r["score"] == "correct") / len(candidate_results)
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"{BASELINE_MODEL} Accuracy: {baseline_accuracy:.2%}")
    print(f"{CANDIDATE_MODEL} Accuracy: {candidate_accuracy:.2%}")
    print(f"Performance Difference: {candidate_accuracy - baseline_accuracy:+.2%}")
    
    if candidate_accuracy > baseline_accuracy:
        print(f"✅ {CANDIDATE_MODEL} outperforms {BASELINE_MODEL}!")
    elif candidate_accuracy == baseline_accuracy:
        print("🤝 Models perform equally well")
    else:
        print(f"❌ {BASELINE_MODEL} outperforms {CANDIDATE_MODEL}")
    
    print("\nDetailed Results:")
    print(f"\n{BASELINE_MODEL} Results:")
    for i, result in enumerate(baseline_results):
        status = "✅" if result["score"] == "correct" else "❌"
        print(f"{status} Case {i+1}: {result['description']} - {result['score']}")
    
    print(f"\n{CANDIDATE_MODEL} Results:")
    for i, result in enumerate(candidate_results):
        status = "✅" if result["score"] == "correct" else "❌"
        print(f"{status} Case {i+1}: {result['description']} - {result['score']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
