import gradio as gr
import pandas as pd
from wikiqa import get_answers_and_context

QUESTION_ANSWER = """\
# Q: {prompt}

## Ground Truth: 

{ground_truth}
"""


def get_data(i=0):
    df = pd.read_csv("test.csv")
    d = df.loc[i].to_dict()

    return d


# get index and perform request in a non blocking way.

# append to csv

with gr.Blocks("Human Evaluation") as demo:
    data = get_data(i=14)

    # factual consistancy
    gr.Markdown("# Factual Consistancy")

    # show context
    context = data["retrieved_context"].split(".")
    context_str = ["## Context"]
    for i, c in enumerate(context):
        context_str.append(f"- {c}")
    gr.Markdown("\n".join(context_str))

    # show question and answer
    gr.Markdown(QUESTION_ANSWER.format(**data))

    for i in range(3):
        with gr.Row():
            with gr.Column(scale=4):
                gr.TextArea(interactive=False)
            with gr.Column(min_width=200):
                gr.Radio(label="Rank", choices=["first", "second", "third"])

    # Relevance
    gr.Markdown("# Relevance")

    with gr.Row():
        gr.Button("clear", variant="secondary")
        gr.Button("submit", variant="primary")

if __name__ == "__main__":
    demo.launch()
