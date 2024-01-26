(get-started-testset-generation)=
# Synthetic test data generation

This tutorial is designed to help you create a synthetic evaluation dataset for assessing your RAG pipeline. To achieve this, we will utilize open-ai models, so please ensure you have your OpenAI API key ready and accessible within your environment.

```{code-block} python
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
```

## Documents

To begin, we require a collection of documents to generate synthetic Question/Context/Answer samples. Here, we will employ the langchain document loader to load documents.

```{code-block} python
:caption: Load documents from directory
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader("your-directory")
documents = loader.load()
```

:::{note}
Each Document object contains a metadata dictionary, which can be used to store additional information about the document which can be accessed with  `Document.metadata`. Please ensure that the metadata dictionary contains a key called `file_name` as this will be used in the generation process. The `file_name` attribute in metadata is used to identify chunks belonging to the same document. For example, pages belonging to the same research publication can be identifies using filename.

An example of how to do this is shown below.

```{code-block} python
for document in documents:
    document.metadata['file_name'] = document.metadata['source']
```
:::

At this point, we have a set of documents at our disposal, which will serve as the basis for creating synthetic Question/Context/Answer triplets.

## Data Generation

We will now import and use Ragas' `Testsetgenerator` to promptly generate a synthetic test set from the loaded documents.

```{code-block} python
:caption: Create 10 samples using default configuration
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# generator with openai models
generator = TestsetGenerator.with_openai()

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
```

Subsequently, we can export the results into a Pandas DataFrame.

```{code-block}
:caption: Export to Pandas
testset.to_pandas()
```
<p align="left">
<img src="../_static/imgs/testset_output.png" alt="test-outputs" width="800" height="600" />
</p>
