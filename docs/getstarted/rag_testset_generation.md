## Testset Generation for RAG

This simple guide will help you generate a testset for evaluating your RAG pipeline using your own documents.

### Load Sample Documents

For the sake of this tutorial we will use sample documents from this [repository](https://huggingface.co/datasets/explodinggradients/Sample_Docs_Markdown). You can replace this with your own documents.

```bash
git clone https://huggingface.co/datasets/explodinggradients/Sample_Docs_Markdown
```

### Load documents

Now we will load the documents from the sample dataset using `DirectoryLoader`, which is one of document loaders from [langchain_community](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/). You may also use any loaders from [llama_index](https://docs.llamaindex.ai/en/stable/understanding/loading/llamahub/)

```python
from langchain_community.document_loaders import DirectoryLoader

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
```

### Choose your LLM

You may choose to use any [LLM of your choice](../howtos/customizations/customize_models.md)
--8<--
choose_evaluvator_llm.md
--8<--

### Generate Testset

Now we will run the test generation using the loaded documents and the LLM setup. If you have used `llama_index` to load documents, please use `generate_with_llama_index_docs` method instead.

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm)
dataset = generator.generate_with_langchain_docs(docs, test_size=10)
```

### Export

You may now export and inspect the generated testset.

```python
dataset.to_pandas()
```
