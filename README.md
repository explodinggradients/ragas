<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="./docs/_static/imgs/logo.png">
</h1>
<p align="center">
  <i>Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines</i>
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
    <a href="https://colab.research.google.com/github/explodinggradients/ragas/blob/main/docs/quickstart.ipynb">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://discord.gg/5djav8GGNZ">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/5djav8GGNZ?style=flat">
    </a>
    <a href="https://github.com/explodinggradients/ragas/">
        <img alt="Downloads" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://docs.ragas.io/">Documentation</a> |
        <a href="#shield-installation">Installation</a> |
        <a href="#fire-quickstart">Quickstart</a> |
        <a href="#-community">Community</a> |
        <a href="#-open-analytics">Open Analytics</a> |
        <a href="https://huggingface.co/explodinggradients">Hugging Face</a>
    <p>
</h4>

> 🚀 Dedicated solutions to evaluate, monitor and improve performance of LLM & RAG application in production including custom models for production quality monitoring.[Talk to founders](https://cal.com/shahul-ragas/30min)

Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG denotes a class of LLM applications that use external data to augment the LLM’s context. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard. This is where Ragas (RAG Assessment) comes in.

Ragas provides you with the tools based on the latest research for evaluating LLM-generated text to give you insights about your RAG pipeline. Ragas can be integrated with your CI/CD to provide continuous checks to ensure performance.

## :shield: Installation

From release:

```bash
pip install ragas
```

Alternatively, from source:

```bash
pip install git+https://github.com/explodinggradients/ragas
```

## :fire: Quickstart

This is a small example program you can run to see ragas in action!

```python

from datasets import Dataset 
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness

os.environ["OPENAI_API_KEY"] = "your-openai-key"

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}

dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset,metrics=[faithfulness,answer_correctness])
score.to_pandas()
```

Refer to our [documentation](https://docs.ragas.io/) to learn more.


## 🫂 Community

If you want to get more involved with Ragas, check out our [discord server](https://discord.gg/5qGUJ6mh7C). It's a fun community where we geek out about LLM, Retrieval, Production issues, and more.

## 🔍 Open Analytics

We track very basic usage metrics to guide us to figure out what our users want, what is working, and what's not. As a young startup, we have to be brutally honest about this which is why we are tracking these metrics. But as an Open Startup, we open-source all the data we collect. You can read more about this [here](https://github.com/explodinggradients/ragas/issues/49). **Ragas does not track any information that can be used to identify you or your company**. You can take a look at exactly what we track in the [code](./src/ragas/_analytics.py)

To disable usage-tracking you set the `RAGAS_DO_NOT_TRACK` flag to true.
