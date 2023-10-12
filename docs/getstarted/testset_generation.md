(get-started-testset-generation)=
# Synthetic test data generation

This tutorial is designed to help you create a synthetic evaluation dataset for assessing your RAG pipeline. To achieve this, we will utilize open-ai models, so please ensure you have your OpenAI API key ready and accessible within your environment.

```{code-block} python
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
```

## Documents

To begin, we require a collection of documents to generate synthetic Question/Context/Answer samples. Here, we will employ the llama-index document loaders to retrieve documents.

```{code-block} python
:caption: Load documents from Semantic Scholar
from llama_index import download_loader

SemanticScholarReader = download_loader("SemanticScholarReader")
loader = SemanticScholarReader()
# Narrow down the search space
query_space = "large language models"
# Increase the limit to obtain more documents
documents = loader.load_data(query=query_space, limit=10)
```

At this point, we have a set of documents at our disposal, which will serve as the basis for creating synthetic Question/Context/Answer triplets.

## Data Generation

We will now import and use Ragas' `Testsetgenerator` to promptly generate a synthetic test set from the loaded documents.

```{code-block} python
:caption: Create 10 samples using default configuration
testsetgenerator = TestsetGenerator.from_default()
test_size = 10
testset = testsetgenerator.generate(documents, test_size=test_size)
```

Subsequently, we can export the results into a Pandas DataFrame.

```{code-block}
:caption: Export to Pandas
testset.to_pandas()
```
<p align="left">
<img src="../_static/imgs/testset_output.png" alt="test-outputs" width="800" height="600" />
</p>
