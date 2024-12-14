<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="./docs/_static/imgs/logo.png">
</h1>
<p align="center">
  <i>Supercharge Your LLM Application Evaluations 🚀</i>
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
        <a href="#fire-quickstart">Quick start</a> |
        <a href="https://discord.gg/5djav8GGNZ">Join Discord</a> |
        <a href="https://blog.ragas.io/">Blog</a> |
        <a href="https://newsletter.ragas.io/">NewsLetter</a> |
        <a href="https://www.ragas.io/careers">Careers</a>
    <p>
</h4>

Objective metrics, intelligent test generation, and data-driven insights for LLM apps

Ragas is your ultimate toolkit for evaluating and optimizing Large Language Model (LLM) applications. Say goodbye to time-consuming, subjective assessments and hello to data-driven, efficient evaluation workflows.
Don't have a test dataset ready? We also do production-aligned test set generation.

## Key Features

- 🎯 Objective Metrics: Evaluate your LLM applications with precision using both LLM-based and traditional metrics.
- 🧪 Test Data Generation: Automatically create comprehensive test datasets covering a wide range of scenarios.
- 🔗 Seamless Integrations: Works flawlessly with popular LLM frameworks like LangChain and major observability tools.
- 📊 Build feedback loops: Leverage production data to continually improve your LLM applications.

## :shield: Installation

Pypi: 

```bash
pip install ragas
```

Alternatively, from source:

```bash
pip install git+https://github.com/explodinggradients/ragas
```

## :fire: Quickstart

### Evaluate your RAG with Ragas metrics

This is 5 main lines:

```python
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
metrics = [LLMContextRecall(), FactualCorrectness(), Faithfulness()]
results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)
```

Find the complete RAG Evaluation Quickstart here: [https://docs.ragas.io/en/latest/getstarted/rag_evaluation/](https://docs.ragas.io/en/latest/getstarted/rag_evaluation/)

<details>
<summary>🖱️Click to see preview of RESULTS</summary>

| user_input | retrieved_contexts | response | reference | context_recall | factual_correctness | faithfulness |
|------------|---------------------|----------|-----------|-----------------|---------------------|---------------|
| What are the global implications of the USA Supreme Court ruling on abortion? | "- In 2022, the USA Supreme Court ... - The ruling has created a chilling effect ..." | The global implications ... Here are some potential implications: | The global implications ... Additionally, the ruling has had an impact beyond national borders ... | 1 | 0.47 | 0.516129 |
| Which companies are the main contributors to GHG emissions ... ? | "- Fossil fuel companies ... - Between 2010 and 2020, human mortality ..." | According to the Carbon Majors database ... Here are the top contributors: | According to the Carbon Majors database ... Additionally, between 2010 and 2020, human mortality ... | 1 | 0.11 | 0.172414 |
| Which private companies in the Americas are the largest GHG emitters ... ? | "The private companies responsible ... The largest emitter amongst state-owned companies ..." | According to the Carbon Majors database, the largest private companies ... | The largest private companies in the Americas ... | 1 | 0.26 | 0 |
</details>

### Generate a test dataset for comprehensive RAG evaluation

What if you don't have the data for folks asking questions when they interact with your RAG system? 

Ragas can help by generating [synthetic test set generation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/) -- where you can seed it with your data and control the difficulty, variety, and complexity. 

## 🫂 Community

If you want to get more involved with Ragas, check out our [discord server](https://discord.gg/5qGUJ6mh7C). It's a fun community where we geek out about LLM, Retrieval, Production issues, and more.

## Contributors

```yml
+----------------------------------------------------------------------------+
|     +----------------------------------------------------------------+     |
|     | Developers: Those who built with `ragas`.                      |     |
|     | (You have `import ragas` somewhere in your project)            |     |
|     |     +----------------------------------------------------+     |     |
|     |     | Contributors: Those who make `ragas` better.       |     |     |
|     |     | (You make PR to this repo)                         |     |     |
|     |     +----------------------------------------------------+     |     |
|     +----------------------------------------------------------------+     |
+----------------------------------------------------------------------------+
```

We welcome contributions from the community! Whether it's bug fixes, feature additions, or documentation improvements, your input is valuable.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 🔍 Open Analytics
At Ragas, we believe in transparency. We collect minimal, anonymized usage data to improve our product and guide our development efforts.

✅ No personal or company-identifying information

✅ Open-source data collection [code](./src/ragas/_analytics.py)

✅ Publicly available aggregated [data](https://github.com/explodinggradients/ragas/issues/49)

To opt-out, set the `RAGAS_DO_NOT_TRACK` environment variable to `true`.
