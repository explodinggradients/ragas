<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="./docs/_static/imgs/logo.png">
</h1>
<p align="center">
  <i>Evaluation library for your LLM applications</i>
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
    <a href="https://pypi.org/project/ragas/">
        <img alt="Open In Colab" src="https://img.shields.io/pypi/dm/ragas">
    </a>
    <a href="https://discord.gg/5djav8GGNZ">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/5djav8GGNZ?style=flat">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://docs.ragas.io/">Documentation</a> |
        <a href="#Quickstart">Quick start</a> |
        <a href="https://dcbadge.vercel.app/api/server/5djav8GGNZ?style=flat">Join Discord</a> |
    <p>
</h4>

[Ragas](https://www.ragas.io/) supercharges your LLM application evaluations with tools to objectively measure performance, simulate test case scenarios, and gain insights by leveraging production data.

## Key Features

- **Metrics**: Different LLM based and non LLM based metrics to objectively evaluate your LLM applications such as RAG, Agents, etc.
- **Test Data Generation**: Synthesize high-quality datasets covering wide variety of scenarios for comprehensive testing of your LLM applications.
- **Integrations**: Seamless integration with all major LLM applications frameworks like langchain and observability tools.

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


- [Run ragas metrics for evaluating RAG](https://docs.ragas.io/en/latest/getstarted/rag_evaluation/)
- [Generate test data for evaluating RAG](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/)

## 🫂 Community

If you want to get more involved with Ragas, check out our [discord server](https://discord.gg/5qGUJ6mh7C). It's a fun community where we geek out about LLM, Retrieval, Production issues, and more.

## 🔍 Open Analytics

We track very basic usage metrics to guide us to figure out what our users want, what is working, and what's not. As a young startup, we have to be brutally honest about this which is why we are tracking these metrics. But as an Open Startup, we open-source all the data we collect. You can read more about this [here](https://github.com/explodinggradients/ragas/issues/49). **Ragas does not track any information that can be used to identify you or your company**. You can take a look at exactly what we track in the [code](./src/ragas/_analytics.py)

To disable usage-tracking you set the `RAGAS_DO_NOT_TRACK` flag to true.
