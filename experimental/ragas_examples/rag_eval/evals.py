from ragas_experimental import Dataset, experiment
from ragas_experimental.metrics import DiscreteMetric
from openai import OpenAI
from ragas_experimental.llms import llm_factory
import os
from .rag import default_rag_client

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
rag_client = default_rag_client(llm_client=openai_client)
llm = llm_factory("openai","gpt-4o", openai_client)

def load_dataset():
    
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )
    
    data_samples = [
    {"question":"What is ragas 0.3", "grading_notes":"- experimentation as the central pillar - provides abstraction for datasets, experiments and metrics - supports evals for RAG, LLM workflows and Agents"},
    {"question":"how are experiment results stored in ragas 0.3?","grading_notes":"- configured using different backends like local, gdrive, etc - stored under experiments/ folder in the backend storage"},
    {"question":"What metrics are supported in ragas 0.3?","grading_notes":"- provides abstraction for discrete, numerical and ranking metrics"},]

    for sample in data_samples:
        row = {"question": sample["question"], "grading_notes": sample["grading_notes"]}
        dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)

@experiment()
async def run_experiment(row):
    response = rag_client.query(row["question"])
    
    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"]
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset) 
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
