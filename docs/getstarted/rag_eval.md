# Evaluate a simple RAG system

The purpose of this guide is to illustrate a simple workflow for testing and evaluating a RAG system with `ragas`. It assumes minimum knowledge in building RAG system and evaluation. Please refer to our [installation instruction](./install.md) for installing `ragas`.

## Basic Setup

We will use `langchain_openai` to set the LLM and embedding model for building our simple RAG. You may choose any other LLM and embedding model of your choice, to do that please refer to [customizing models in langchain](https://python.langchain.com/docs/integrations/chat/).


```python
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
```

### Build a Simple RAG System

To build a simple RAG system, we need to define the following components:

- Define a method to vectorize our docs
- Define a method to retrieve the relevant docs
- Define a method to generate the response

??? note "Click to View the Code"

    ```python

    import numpy as np

    class RAG:
        def __init__(self, model="gpt-4o"):
            self.llm = ChatOpenAI(model=model)
            self.embeddings = OpenAIEmbeddings()
            self.doc_embeddings = None
            self.docs = None

        def load_documents(self, documents):
            """Load documents and compute their embeddings."""
            self.docs = documents
            self.doc_embeddings = self.embeddings.embed_documents(documents)

        def get_most_relevant_docs(self, query):
            """Find the most relevant document for a given query."""
            if not self.docs or not self.doc_embeddings:
                raise ValueError("Documents and their embeddings are not loaded.")
            
            query_embedding = self.embeddings.embed_query(query)
            similarities = [
                np.dot(query_embedding, doc_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
                for doc_emb in self.doc_embeddings
            ]
            most_relevant_doc_index = np.argmax(similarities)
            return [self.docs[most_relevant_doc_index]]

        def generate_answer(self, query, relevant_doc):
            """Generate an answer for a given query based on the most relevant document."""
            prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
            messages = [
                ("system", "You are a helpful assistant that answers questions based on given documents only."),
                ("human", prompt),
            ]
            ai_msg = self.llm.invoke(messages)
            return ai_msg.content
    ```

### Load Documents
Now, let's load some documents and test our RAG system.

```python
sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]
```

```python
# Initialize RAG instance
rag = RAG()

# Load documents
rag.load_documents(sample_docs)

# Query and retrieve the most relevant document
query = "Who introduced the theory of relativity?"
relevant_doc = rag.get_most_relevant_docs(query)

# Generate an answer
answer = rag.generate_answer(query, relevant_doc)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")
```


Output:
```
Query: Who introduced the theory of relativity?
Relevant Document: ['Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.']
Answer: Albert Einstein introduced the theory of relativity.
```

## Collect Evaluation Data

To collect evaluation data, we first need a set of queries to run against our RAG. We can run the queries through the RAG system and collect the `response`, `retrieved_contexts`for each query. You may also optionally prepare a set of golden answers for each query to evaluate the system's performance.



```python


sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]
```

```python
dataset = []

for query,reference in zip(sample_queries,expected_responses):
    
    relevant_docs = rag.get_most_relevant_docs(query)
    response = rag.generate_answer(query, relevant_docs)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )
```

Now, load the dataset into `EvaluationDataset` object.

```python
from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)
```

## Evaluate

We have successfully collected the evaluation data. Now, we can evaluate our RAG system on the collected dataset using a set of commonly used RAG evaluation metrics. You may choose any model as [evaluator LLM](./../howtos/customizations/customize_models.md) for evaluation. 

```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper


evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
result
```

Output
```
{'context_recall': 1.0000, 'faithfulness': 0.8571, 'factual_correctness': 0.7280}
```

## Analyze Results

Once you have evaluated, you may want to view, analyse and share results. This is important to interpret the results and understand the performance of your RAG system. For this you may sign up and setup [app.ragas.io]() easily. If not, you may use any alternative tools available to you. 

In order to use the [app.ragas.io](http://app.ragas.io) dashboard, you need to have an account on [app.ragas.io](https://app.ragas.io/). If you don't have one, you can sign up for one [here](https://app.ragas.io/login). You will also need to generate a [Ragas APP token](https://app.ragas.io/dashboard/settings/app-tokens).

Once you have the API key, you can use the `upload()` method to export the results to the dashboard.

```python
import os
os.environ["RAGAS_APP_TOKEN"] = "your_app_token"
```

Now you can view the results in the dashboard by following the link in the output of the `upload()` method.

```python
result.upload()
```

![](rag_eval.gif)

## Up Next

- [Generate test data for evaluating RAG](rag_testset_generation.md)