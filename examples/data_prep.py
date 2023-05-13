from datasets import concatenate_datasets, load_dataset


def format_for_belar(row):
    row["context"] = row["selftext"]
    row["prompt"] = row["title"]
    row["ground_truth"] = row["answers"]["text"]
    return row


d = load_dataset("eli5")
ds = d["test_eli5"].map(format_for_belar, batched=False)
ds = ds.select_columns(["context", "prompt", "ground_truth"])

ds = ds.shuffle(seed=42).select(range(500))
ds.shape, ds.column_names

import concurrent.futures as f

from langchain.llms import OpenAI

llm = OpenAI()
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
