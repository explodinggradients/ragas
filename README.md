<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="./docs/assets/logo.png">
</h1>
<p align="center">
  <i>SOTA metrics for evaluating Retrieval Augmented Generation (RAG)</i>
</p>

<p align="center">
    <a href="https://github.com/explodinggradients/ragas/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/explodinggradients/ragas.svg">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/explodinggradients/ragas/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/explodinggradients/ragas.svg?color=green">
    </a>
    <a href="https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://github.com/explodinggradients/ragas/">
        <img alt="Downloads" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#shield-installation">Installation</a> |
        <a href="#fire-quickstart">Quickstart</a> |
        <a href="#luggage-metrics">Metrics</a> |
        <a href="#raising_hand_man-faq">FAQ</a> |
        <a href="https://huggingface.co/explodinggradients">Hugging Face</a>
    <p>
</h4>

ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG denotes a class of LLM applications that use external data to augment the LLM’s context. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard.. This is were ragas (RAG Assessment) comes in

ragas provides you with the tools based on the latest research for evaluating LLM generated text  to give you insights about your RAG pipeline. ragas can be integrated with your CI/CD to provide continuous check to ensure performance.

## :shield: Installation

```bash
pip install ragas
```
if you want to install from source 
```bash
git clone https://github.com/explodinggradients/ragas && cd ragas
pip install -e .
```

## :fire: Quickstart 

This is a small example program you can run to see ragas in action!
```python

from ragas import evaluate
from datasets import Dataset
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"

# prepare your huggingface dataset in the format
# Dataset({
#     features: ['question','contexts','answer'],
#     num_rows: 25
# })

dataset: Dataset

results = evaluate(dataset)

```
If you want a more in-depth explanation of core components, check out our [quick-start notebook](./examples/quickstart.ipynb)
## :luggage: Metrics

Ragas measures your pipeline's performance against two dimensions
1. **Factuality**: measures the factual consistency of the generated answer against the given context.
2. **Relevancy**:  measures how relevant retrieved contexts and the generated answer are to the question. 

Through repeated experiments, we have found that the quality of a RAG pipeline is highly dependent on these two dimensions. The final `ragas_score` is the harmonic mean of these two factors. 

To read more about our metrics, checkout [docs](/docs/metrics.md).
## :question: How to use Ragas to improve your pipeline?
*"Measurement is the first step that leads to control and eventually to improvement" - James Harrington*

Here we assume that you already have your RAG pipeline ready. When is comes to RAG pipelines, there are mainly two parts - Retriever and generator. A change in any of this should also impact your pipelines's quality.

1. First, decide one parameter that you're interested in adjusting. for example the number of retrieved documents, K. 
2. Collect a set of sample prompts (min 20) to form your test set.
3. Run your pipeline using the test set before and after the change. Each time record the prompts with context and generated output.
4. Run ragas evaluation for each of them to generate evaluation scores. 
5. Compare the scores and you will know how much the change has affected your pipelines's performance.


## :raising_hand_man: FAQ
1. Why harmonic mean?
Harmonic mean penalizes extreme values. For example if your generated answer is fully factually consistent with the context (factuality = 1) but is not relevant to the question (relevancy = 0), simple average would give you a score of 0.5 but harmonic mean will give you 0.0




 


