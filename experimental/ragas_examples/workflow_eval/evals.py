import os
from openai import OpenAI
from ragas_experimental import Dataset, experiment
from ragas_experimental.metrics import DiscreteMetric
from ragas_experimental.llms import llm_factory
from .workflow import default_workflow_client


workflow_client = default_workflow_client()
llm = llm_factory("openai", "gpt-4o", OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))




def load_dataset():

    dataset_dict = [
    {
        "email": "Hi, I'm getting error code XYZ-123 when using version 2.1.4 of your software. Please help!",
        "pass_criteria": "category Bug Report; product_version 2.1.4; error_code XYZ-123; response references both version and error code"
    },
    
    {
        "email": "I need to dispute invoice #INV-2024-001 for 299.99 dollars. The charge seems incorrect.",
        "pass_criteria": "category Billing; invoice_number INV-2024-001; amount 299.99; response references invoice and dispute process"
    },
    
    {
        "email": "Would love to see a dark mode feature in the dashboard. This is really important for our team!",
        "pass_criteria": "category Feature Request; requested_feature dark mode; product_area dashboard; urgency_level high/medium; response acknowledges dark mode request"
    },
    
    {
        "email": "The system crashes with ERR_MEMORY_OVERFLOW but I can't find the version number anywhere.",
        "pass_criteria": "category Bug Report; error_code ERR_MEMORY_OVERFLOW; product_version null; response handles missing version gracefully"
    },
    
    {
        "email": "Please add the ability to export reports as PDF files. This is urgent for our quarterly review.",
        "pass_criteria": "category Feature Request; requested_feature export PDF; product_area reports; urgency_level urgent/high; response reflects urgency"
    },
    
    {
        "email": "It would cool to have a feature that allows users to customize their dashboard layout.",
        "pass_criteria": "category Feature Request; requested_feature customize dashboard; product_area dashboard; urgency_level low/medium; response matches casual tone"
    },
    
    {
        "email": "I am getting an error when I try to access the API. The error code is API-500 and I am using the latest version of the SDK.",
        "pass_criteria": "category Bug Report; error_code API-500; product_version latest/null; response acknowledges API context and vague version"
    },
    
    {
        "email": "The application crashed on me. I'm running v2.5.1-beta and got this weird message: 'FATAL_ERROR_001'. Can you help?",
        "pass_criteria": "category Bug Report; product_version 2.5.1-beta; error_code FATAL_ERROR_001; response handles beta version and crash"
    },
    
    {
        "email": "I was charged 1,299 dollars but my invoice number is BILL2024-March-001. This seems wrong.",
        "pass_criteria": "category Billing; invoice_number BILL2024-March-001; amount 1299; response handles non-standard formats"
    },
    
    {
        "email": "Feature needed:Real-time sync,Area:Mobile app,Priority:HIGH",
        "pass_criteria": "category Feature Request; requested_feature Real-time sync; product_area mobile; urgency_level high; response parses structured format"
    }]
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )
    for sample in dataset_dict:
        row = {"email": sample["email"], "pass_criteria": sample["pass_criteria"]}
        dataset.append(row)
        
    dataset.save()  # Save the dataset
    return dataset


my_metric = DiscreteMetric(
    name="response_quality",
    prompt="Evaluate the response based on the pass criteria: {pass_criteria}. Does the response meet the criteria? Return 'pass' or 'fail'.\nResponse: {response}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    response = workflow_client.process_email(
        row["email"]
    )
    
    score = my_metric.score(
        llm=llm,
        response=response.get("response_template", " "),
        pass_criteria=row["pass_criteria"]
    )

    experiment_view = {
        **row,
        "response": response.get("response_template", " "),
        "score": score.value,
        "score_reason": score.reason,
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    _ = await run_experiment.arun(dataset)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())