# LangChain Integration

This tutorial demonstrates how to evaluate a RAG-based Q&A application built with LangChain using Ragas. Additionally, we will explore how the Ragas App can help analyze and enhance the application's performance.

### Building a simple Q&A application

To build a question-answering system, we start by creating a small dataset and indexing it using its embeddings in a vector database.


```python
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

content_list = [
    "Andrew Ng is the CEO of Landing AI and is known for his pioneering work in deep learning. He is also widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He is a strong advocate for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems that can master complex games like AlphaGo.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., and he is praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences on a global scale.",
    "Arvind Krishna is the CEO of IBM and is recognized for transforming the company towards cloud computing and AI solutions. He focuses on providing cutting-edge technologies to address modern business challenges.",
]

langchain_documents = []

for content in content_list:
    langchain_documents.append(
        Document(
            page_content=content,
        )
    )
```


```python
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

_ = vector_store.add_documents(langchain_documents)
```

We will now build a RAG-based system that integrates the retriever, LLM, and prompt into a Retrieval QA Chain. The retriever fetches relevant documents from a knowledge base. LLM will generate responses based on the retrieved documents using the Prompt which will guide the model's response, helping it understand the context and generate relevant and coherent language-based output.

In LangChain, we can create a retriever from a vector store by using its `.as_retriever` method. For more details, refer to the [LangChain documentation on vector store retrievers](https://python.langchain.com/docs/how_to/vectorstore_retriever/).


```python
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
```


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

We will define a Chain that processes the user query and retrieved relevant data, passing it to the model within a structured prompt. The model's output is then parsed to generate the final response as a string.


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


template = """Answer the question based only on the following context:
{context}

Question: {query}
"""
prompt = ChatPromptTemplate.from_template(template)

qa_chain = prompt | llm | StrOutputParser()
```


```python
def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


query = "Who is the CEO of OpenAI?"

relevant_docs = retriever.invoke(query)
qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
```
Output:
```
'The CEO of OpenAI is Sam Altman.'
```


### Evaluate


```python
sample_queries = [
    "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
    "Who is Sam Altman?",
    "Who is Demis Hassabis and how did he gained prominence?",
    "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
    "How did Arvind Krishna transformed IBM?",
]

expected_responses = [
    "Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally.",
    "Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges.",
]
```

To evaluate the Q&A system we need to structure the queries, expected_responses and other metric secpific requirments to [EvaluationDataset][ragas.dataset_schema.EvaluationDataset].


```python
from ragas import EvaluationDataset


dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)
```

To evauate our Q&A application we will use the following metrices.  


- `LLMContextRecall`: Evaluates how well retrieved contexts align with claims in the reference answer, estimating recall without manual reference context annotations.
- `Faithfulness`: Assesses whether all claims in the generated answer can be inferred directly from the provided context.
- `Factual Correctness`: Checks the factual accuracy of the generated response by comparing it with a reference, using claim-based evaluation and natural language inference.  

For more details on these metrics and how they apply to evaluating RAG systems, visit [Ragas Metrics Documentation](./../../concepts/metrics/available_metrics/).


```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)

result
```

Output
```
{'context_recall': 1.0000, 'faithfulness': 0.9000, 'factual_correctness': 0.9260}
```