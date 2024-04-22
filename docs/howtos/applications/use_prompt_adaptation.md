# Automatic language adaptation

1. [Metrics](#language-adaptation-for-metrics)
2. [Testset generation](#language-adaptation-for-testset-generation)

## Language Adaptation for Metrics

This is a tutorial notebook showcasing how to successfully use ragas with data from any given language. This is achieved using Ragas prompt adaptation feature. The tutorial specifically applies ragas metrics to a Hindi RAG evaluation dataset.

### Dataset
Here I’m using a dataset containing all the relevant columns in Hindi language. 

```{code-block} python

from datasets import load_dataset, Dataset

hindi_dataset = load_dataset("explodinggradients/amnesty_qa","hindi")
hindi_dataset
```

```{code-block}
DatasetDict({
    train: Dataset({
        features: ['question', 'ground_truth', 'answer', 'contexts'],
        num_rows: 20
    })
})
```

### Adapt metrics to target language

Import any metrics from Ragas as required and adapt and save each one of them to the target language using `adapt` function. Optionally you can also specify which llm to use for prompt adaptation, here I am using `gpt-4`. It is highly recommended to use the best llm here as quality of adapted prompts highly influence the results. 

```{code-block} python

from ragas.metrics import (
    faithfulness,
    answer_correctness,
)
from langchain.chat_models import ChatOpenAI
from ragas import adapt

# llm used for adaptation
openai_model = ChatOpenAI(model_name="gpt-4")

adapt(metrics=[faithfulness,answer_correctness], language="hindi", llm=openai_model)
```

The prompts belonging to respective metrics will be now automatically adapted to the target language. The save step saves it to `.cacha/ragas` by default to reuse later.  Next time when you do adapt with the same metrics, ragas first checks if the adapted prompt is already present in the cache. 

Let’s inspect the adapted prompt belonging to the answer correctness metric

```{note}
When adapting prompts, it is recommended to review them manually prior to evaluation, as language models may introduce errors during translation
````


```{code-block} python
print(answer_correctness.correctness_prompt.to_string())
```
```{code-block}
Extract the following from the given question and the ground truth

question: "सूरज को क्या चलाता है और इसका प्राथमिक कार्य क्या है?"
answer: "सूर्य परमाणु विघटन द्वारा संचालित होता है, जो पृथ्वी पर परमाणु रिएक्टरों के समान होते हैं, और इसका प्राथमिक कार्य सौरमंडल को प्रकाश प्रदान करना है।"
ground_truth: "सूर्य वास्तव में परमाणु संयोजन द्वारा चलाया जाता है, न कि विखंडन द्वारा। इसके केंद्र में, हाइड्रोजन परमाणु हीलियम बनाने के लिए मिल जाते हैं, जिससे बहुत अधिक ऊर्जा मुक्त होती है। यह ऊर्जा ही सूर्य को जलाती है और जीवन के लिए महत्वपूर्ण ताप और प्रकाश प्रदान करती है। सूर्य का प्रकाश भूमि की जलवायु प्रणाली में भी महत्वपूर्ण भूमिका निभाता है और मौसम और समुद्री धाराओं को चलाने में मदद करता है।"
Extracted statements: [{{"statements that are present in both the answer and the ground truth": ["सूर्य का प्राथमिक कार्य प्रकाश प्रदान करना है"], "statements present in the answer but not found in the ground truth": ["सूर्य पारमाणु विखंडन द्वारा संचालित होता है", "पृथ्वी पर पारमाणु रिएक्टरों के समान"], "relevant statements found in the ground truth but omitted in the answer": ["सूर्य पारमाणु संयोजन द्वारा संचालित होता है, न कि विखंडन", "इसके कोर में, हाइड्रोजन परमाणु हीलियम बनाने के लिए मिलकर तेजी से जलते हैं, जिससे बहुतायत ऊर्जा मुक्त होती है", "यह ऊर्जा जीवन के लिए आवश्यक गर्मी और प्रकाश प्रदान करती है", "सूर्य का प्रकाश पृथ्वी की जलवायु प्रणाली में महत्वपूर्ण भूमिका निभाता है", "सूर्य मौसम और समुद्री धाराओं को चलाने में मदद करता है"]}}]

question: "पानी का उबलने का बिंदु क्या है?"
answer: "पानी का उबलने का बिंदु समुद्री स्तर पर 100 डिग्री सेल्सियस है।"
ground_truth: "पानी का उबलने का बिंदु समुद्र तल पर 100 डिग्री सेल्सियस (212 डिग्री फारेनहाइट) होता है, लेकिन यह ऊचाई के साथ बदल सकता है।"
Extracted statements: [{{"statements that are present in both the answer and the ground truth": ["पानी का उबलने का बिंदु समुद्री स्तर पर 100 डिग्री सेल्सियस पर होता है"], "statements present in the answer but not found in the ground truth": [], "relevant statements found in the ground truth but omitted in the answer": ["ऊचाई के साथ उबलने का बिंदु बदल सकता है", "पानी का उबलने का बिंदु समुद्री स्तर पर 212 डिग्री फारेनहाइट होता है"]}}]

question: {question}
answer: {answer}
ground_truth: {ground_truth}
Extracted statements:
```

The instruction and key objects are kept unchanged intentionally to allow consuming and processing results in ragas.  During inspection, if any of the demonstrations seem faulty translated you can always correct it by going to the saved location. 

### Evaluate

```{code-block} python
from ragas import evaluate

ragas_score = evaluate(dataset['train'], metrics=[faithfulness,answer_correctness])
```

You will observe much better performance now with Hindi language as prompts are tailored to it.


## Language Adaptation for Testset Generation

This is a tutorial notebook showcasing how to successfully use ragas test data generation feature to generate data samples of any language using list of documents. This is achieved using Ragas prompt adaptation feature. The tutorial specifically applies ragas test set generation to a Hindi to produce a question answer dataset in Hindi.

### Documents
Here I'm using a corpus of wikipedia articles written in Hindi. You can download the articles by 


```{code-block} bash
git lfs install
git clone https://huggingface.co/datasets/explodinggradients/hindi-wikipedia
```

Now you can load the documents using a document loader, here I am using `DirectoryLoader`

```{code-block} python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./hindi-wikipedia/")
documents = loader.load()

# add metadata
for document in documents:
    document.metadata['filename'] = document.metadata['source']

```

### Import and adapt evolutions
Now we can import all the required evolutions and adapt it using `generator.adapt`. This will also adapt all the necessary filters required for the corresponding evolutions. Once adapted, it's better to save and inspect the adapted prompts. 


```{code-block} python

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context,conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# adapt to language
language = "hindi"

generator.adapt(language, evolutions=[simple, reasoning,conditional,multi_context])
generator.save(evolutions=[simple, reasoning, multi_context,conditional])
```

### Generate dataset
Once adapted you can use the evolutions and generator just like before to generate data samples for any given distribution.

```{code-block} python
# determine distribution

distributions = {
    simple:0.4,
    reasoning:0.2,
    multi_context:0.2,
    conditional:0.2
    }


# generate testset
testset = generator.generate_with_langchain_docs(documents, 10,distributions,with_debugging_logs=True)
testset.to_pandas()
```
