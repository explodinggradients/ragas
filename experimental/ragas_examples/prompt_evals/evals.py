from ragas_experimental import Dataset, experiment
from ragas_experimental.metrics.result import MetricResult
from ragas_experimental.metrics.discrete import discrete_metric

from .prompt import run_prompt

@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def my_metric(prediction: str, actual: str):
    """Calculate accuracy of the prediction."""
    return MetricResult(value="pass", reason="") if prediction == actual else MetricResult(value="fail", reason="")
    
    
@experiment()
async def run_experiment(row):
    
    response = run_prompt(row["text"])
    score = my_metric.score(
        prediction=response,
        actual=row["label"]
    )

    experiment_view = {
        **row,
        "response":response,
        "score":score.value,
    }
    return experiment_view


def load_dataset():
    # Create a dataset
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )
    dataset_dict = [
    {"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    {"text": "It was an average film, nothing special.", "label": "positive"},
    {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"},
    {"text": "I did not like it at all, very disappointing.", "label": "negative"},
    {"text": "It was okay, not the best but not the worst.", "label": "positive"},
    {"text": "I have mixed feelings about it, some parts were good, others not so much.", "label": "positive"},
    {"text": "What a masterpiece! I would watch it again.", "label": "positive"},
    {"text": "I would not recommend it to anyone, it was that bad.", "label": "negative"},]

    for sample in dataset_dict:
        row = {"text":sample["text"], "label":sample["label"]}
        dataset.append(row)

    # make sure to save it
    dataset.save() 
    return dataset


async def main():
    dataset = load_dataset()
    experiment_results = await run_experiment.arun(dataset) 
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)
    
   


if __name__ == "__main__":
    
    import asyncio
    asyncio.run(main())