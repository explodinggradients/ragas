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

[Ragas](https://www.ragas.io/) supercharges your LLM application evaluations with tools to objectively measure performance, synthesize test case scenarios, and gain insights by leveraging production data.

Evaluating and testing LLM applications is a challenging, time-consuming, and often boring process. Ragas aims provide a suite of tools that could supercharge your evaluation workflows and make it more efficient and fun using  state-of-the-art research. We are also building an open ecosystem, that fosters sharing of ideas to make the evaluation process better and collaborates with other tools in the market to make it a seamless experience.

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
At Ragas, we believe in transparency. We collect minimal, anonymized usage data to improve our product and guide our development efforts.

✅ No personal or company-identifying information
✅ Open-source data collection [code](./src/ragas/_analytics.py)
✅ Publicly available aggregated [data](https://github.com/explodinggradients/ragas/issues/49)

To opt-out, set the `RAGAS_DO_NOT_TRACK` environment variable to `true`.
