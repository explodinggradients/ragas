from ragas import Dataset, experiment
from ragas.metrics.numeric import numeric_metric
from ragas.metrics.result import MetricResult

from .agent import get_default_agent

math_agent = get_default_agent()


@numeric_metric(name="correctness", allowed_values=(0.0, 1.0))
def correctness_metric(prediction: float, actual: float):
    """Calculate correctness of the prediction."""
    if isinstance(prediction, str) and "ERROR" in prediction:
        return 0.0
    result = 1.0 if abs(prediction - actual) < 1e-5 else 0.0
    return MetricResult(
        value=result, reason=f"Prediction: {prediction}, Actual: {actual}"
    )


def load_dataset():
    # Create a dataset
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )
    # Create sample data for mathematical expressions and their results
    math_problems = [
        {"question": "15 - 3 / 4", "answer": 14.25},
        {"question": "(2 + 3) * (6 - 2)", "answer": 20.0},
        {"question": "100 / 5 + 3 * 2", "answer": 26.0},
        {"question": "((2 * 3) + (4 * 5)) * ((6 - 2) / (8 / 4))", "answer": 52.0},
        {"question": "2 + 3 * 4 - 5 / 6 + 7", "answer": 20.166666666666664},
        {"question": "(10 / 2) + (20 / 4) + (30 / 6) + (40 / 8)", "answer": 20.0},
        {"question": "1/3 + 1/3 + 1/3", "answer": 1.0},
    ]

    # Add the data to the dataset
    for row in math_problems:
        dataset.append(row)

    dataset.save()  # Save the dataset
    return dataset


@experiment()
async def run_experiment(row):
    question = row["question"]
    expected_answer = row["answer"]

    # Get the model's prediction
    prediction = math_agent.solve(question)

    # Calculate the correctness metric
    correctness = correctness_metric.score(
        prediction=prediction.get("result"), actual=expected_answer
    )

    return {
        "question": question,
        "expected_answer": expected_answer,
        "prediction": prediction.get("result"),
        "log_file": prediction.get("log_file"),
        "correctness": correctness.value,
    }


async def main():
    dataset = load_dataset()
    experiment_result = await run_experiment.arun(dataset)
    print("Experiment_result: ", experiment_result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
