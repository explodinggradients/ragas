import concurrent.futures as f

from datasets import DatasetDict, load_dataset
from langchain.llms import OpenAI


def format_for_belar(row):
    row["context"] = row["selftext"]
    row["prompt"] = row["title"]
    row["ground_truth"] = row["answers"]["text"]
    return row


d = load_dataset("eli5")
assert isinstance(d, DatasetDict)
ds = d["test_eli5"].map(format_for_belar, batched=False)
ds = ds.select_columns(["context", "prompt", "ground_truth"])

ds = ds.shuffle(seed=42).select(range(500))
print(ds.shape, ds.column_names)


llm = OpenAI()  # type: ignore
prompt = """
{context}
with the above context explain like I'm five: {prompt}
"""


def get_answers(row):
    qs, cs = row["prompt"], row["context"]

    generated_answers = []
    with f.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(
            llm, [prompt.format(context=cs[i], prompt=qs[i]) for i in range(len(qs))]
        )
        for result in results:
            generated_answers.append(result)

    row["generated_answers"] = generated_answers
    return row


ds = ds.map(get_answers, batched=True, batch_size=10)
