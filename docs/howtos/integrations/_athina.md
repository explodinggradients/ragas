# Athina AI
## Ragas Metrics on your Production Logs

[Athina](https://athina.ai) is a production monitoring and evaluation platform. Try the [sandbox](https://demo.athina.ai/observe?filters=dateSpan%3D30) here.

You can use [Athina with Ragas](http://localhost:3001/evals/preset_evals/ragas_evals) metrics to run evals on production logs, and get granular model performance metrics on your production data.

![Athina Performance Metrics](https://docs.athina.ai/performance-metrics.png)

For example, you can get insights like this visually:
- What is my `AnswerRelevancy` score for queries related to `refunds` for customer id `nike-usa`
- What is my `Faithfulness` score for `product catalog` queries using prompt `catalog_answerer/v3` with model `gpt-3.5-turbo`

### ▷ Running Athina Programmatically

When you use Athina to run Ragas evals programmatically, you will be able to view the results on Athina's UI like this 👇

![View RAGAS Metrics on Athina](https://docs.athina.ai/ragas-develop-view.png)

1. Install Athina's Python SDK:

```
pip install athina
```

2. Create an account at [app.athina.ai](https://app.athina.ai). After signing up, you will receive an API key.

Here's a sample notebook you can follow: https://github.com/athina-ai/athina-evals/blob/main/examples/ragas.ipynb

3. Run the code


```python
import os
from athina.evals import (
    RagasAnswerCorrectness,
    RagasAnswerRelevancy,
    RagasContextRelevancy,
    RagasFaithfulness,
)
from athina.loaders import RagasLoader
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.runner.run import EvalRunner
import pandas as pd

# Set your API keys
OpenAiApiKey.set_key(os.getenv("OPENAI_API_KEY"))
AthinaApiKey.set_key(os.getenv("ATHINA_API_KEY"))

# Load your dataset from a dictionary, json, or csv: https://docs.athina.ai/evals/loading_data
dataset = RagasLoader().load_json("raw_data.json")

# Configure the eval suite
eval_model = "gpt-3.5-turbo"
eval_suite = [
    RagasAnswerCorrectness(),
    RagasFaithfulness(),
    RagasContextRelevancy(),
    RagasAnswerRelevancy(),
]

# Run the evaluation suite
batch_eval_result = EvalRunner.run_suite(
    evals=eval_suite,
    data=dataset,
    max_parallel_evals=1,  # If you increase this, you may run into rate limits
)

pd.DataFrame(batch_eval_result)
```

### ▷ Configure Ragas to run automatically on your production logs

If you are [logging your production inferences to Athina](https://docs.athina.ai/logging/log_via_api), you can configure Ragas metrics to run automatically against your production logs.

1. Navigate to the [Athina Dashboard](https://app.athina.ai/evals/config)
   
2. Open the **Evals** page (lightning icon on the left)
3. Click the "New Eval" button on the top right
4. Select the **Ragas** tab
5. Select the eval you want to configure

![Set up Ragas on Athina UI](https://docs.athina.ai/ragas-modal-bg.png)

#### Learn more about Athina
- **Website:** [https://athina.ai](https://athina.ai)
- **Docs:** [https://docs.athina.ai](https://docs.athina.ai)
- **Github Library:** [https://github.com/athina-ai/athina-evals](https://github.com/athina-ai/athina-evals)
- **Sandbox**: [https://demo.athina.ai](https://demo.athina.ai/observe?filters=dateSpan%3D30)
