(testset-generation)=
# Synthetic Test Data generation 

## Why synthetic test data?

Evaluating RAG (Retrieval-Augmented Generation) augmented pipelines is crucial for assessing their performance. However, manually creating hundreds of QA (Question-Context-Answer) samples from documents can be time-consuming and labor-intensive. Additionally, human-generated questions may struggle to reach the level of complexity required for a thorough evaluation, ultimately impacting the quality of the assessment. **By using synthetic data generation developer time in data aggregation process can be reduced by 90%**. 

## How does Ragas differ in test data generation?

Ragas takes a novel approach to evaluation data generation. An ideal evaluation dataset should encompass various types of questions encountered in production, including questions of varying difficulty levels. LLMs by default are not good at creating diverse samples as it tends to follow common paths. Inspired by works like [Evol-Instruct](https://arxiv.org/abs/2304.12244), Ragas achieves this by employing an evolutionary generation paradigm, where **questions with different characteristics such as reasoning, conditioning, multi-context, and more are systematically crafted from the provided set of documents**. This approach ensures comprehensive coverage of the performance of various components within your pipeline, resulting in a more robust evaluation process.

<p align="center">
<img src="../_static/imgs/eval-evolve.png" alt="evol-generate" width="600" height="400" />
</p>


### In-Depth Evolution

Language Language Models (LLMs) possess the capability to transform simple questions into more complex ones effectively. To generate medium to hard samples from the provided documents, we employ the following methods:

- **Reasoning:** Rewrite the question in a way that enhances the need for reasoning to answer it effectively.

- **Conditioning:** Modify the question to introduce a conditional element, which adds complexity to the question.

- **Multi-Context:** Rephrase the question in a manner that necessitates information from multiple related sections or chunks to formulate an answer.

Moreover, our paradigm extends its capabilities to create conversational questions from the given documents:

- **Conversational:** A portion of the questions, following the evolution process, can be transformed into conversational samples. These questions simulate a chat-based question-and-follow-up interaction, mimicking a chat-Q&A pipeline.

```{note}
Moving forward, we are will be expanding the range of evolution techniques to offer even more diverse evaluation possibilities
```


## Example

```{code-block} python
:caption: Customising test set generation 
from ragas.testset import TestsetGenerator
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


# documents = load your documents

# Add custom llms and embeddings
generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings_model = OpenAIEmbeddings()

# Change resulting question type distribution
testset_distribution = {
    "simple": 0.25,
    "reasoning": 0.5,
    "multi_context": 0.0,
    "conditional": 0.25,
}

# percentage of conversational question
chat_qa = 0.2


test_generator = TestsetGenerator(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings_model=embeddings_model,
    testset_distribution=testset_distribution,
    chat_qa=chat_qa,
)

testset = testsetgenerator.generate(documents, test_size=100)

```

```{code-block} python 
:caption: Export the results into pandas
test_df = testset.to_pandas()
test_df.head()
```

<p align="left">
<img src="../_static/imgs/testset_output.png" alt="test-outputs" width="800" height="600" />
</p>


## Analyze question types

 Analyze the frequency of different question types in the created dataset

 ```{code-block} python
 :caption: bar graph of question types
import seaborn as sns
sns.set(rc={'figure.figsize':(9,6)})

test_data_dist = test_df.question_type.value_counts().to_frame().reset_index()
sns.set_theme(style="whitegrid")
g = sns.barplot(y='count',x='question_type', data=test_data_dist)
g.set_title("Question type distribution",fontdict = { 'fontsize': 20})
 ```

<p align="left">
<img src="../_static/imgs/question_types.png" alt="test-outputs" width="450" height="400" />
</p>
