# Explain or debug LLM based metrics using tracing

While evaluating using LLM based metrics, each metric may make one or more calls to the LLM. These traces are important to understand the results of the metrics and to debug any issues.
This notebook demonstrates how to export the LLM traces and analyze them.

## Evaluation
Do a sample evaluation using one of the LLM based metrics.


```python
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.metrics._aspect_critic import AspectCriticWithReference

dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")


eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

metric = AspectCriticWithReference(
    name="answer_correctness",
    definition="is the response correct compared to reference",
)

results = evaluate(eval_dataset[:5], metrics=[metric])
```

    Repo card metadata block was not found. Setting CardData to empty.
    Evaluating: 100%|██████████| 5/5 [00:02<00:00,  2.12it/s]


## Export LLM traces


```python
results.traces
```




    [{'answer_correctness': 1},
     {'answer_correctness': 0},
     {'answer_correctness': 0},
     {'answer_correctness': 0},
     {'answer_correctness': 0}]



Each of these are [`MetricTrace`](ragas.callbacks.MetricTrace) objects that contain the following fields:
- The input to the prompt 
- The output from the model

Both as pydantic objects.

To view this you can select the index of the trace you want to view and run the cell below.


```python
results.traces[0]["answer_correctness"]
```




    {'0_single_turn_aspect_critic_prompt_with_reference': {'input': AspectCriticInputWithReference(user_input="`user_input`: What are the global implications of the USA Supreme Court ruling on abortion? Answer using `retrieved context`: - In 2022, the USA Supreme Court handed down a decision ruling that overturned 50 years of jurisprudence recognizing a constitutional right to abortion.\n- This decision has had a massive impact: one in three women and girls of reproductive age now live in states where abortion access is either totally or near-totally inaccessible.\n- The states with the most restrictive abortion laws have the weakest maternal health support, higher maternal death rates, and higher child poverty rates.\n- The USA Supreme Court ruling has also had impacts beyond national borders due to the geopolitical and cultural influence wielded by the USA globally and the aid it funds.\n- SRR organizations and activists across the world have expressed fear about the ruling laying the groundwork for anti-abortion legislative and policy attacks in other countries.\n- Advocates have also observed the ruling's impact on progressive law reform and the stalling of the adoption and enforcement of abortion guidelines in certain African countries.\n- The ruling has created a chilling effect in international policy spaces, emboldening anti-abortion state and non-state actors to undermine human rights protections.\nThe USA Supreme Court ruling on abortion has sparked intense debates and discussions not only within the country but also around the world. Many countries look to the United States as a leader in legal and social issues, so the decision could potentially influence the policies and attitudes towards abortion in other nations.\nThe ruling may also impact international organizations and non-governmental groups that work on reproductive rights and women's health issues. Depending on the outcome, there could be shifts in funding, advocacy efforts, and collaborations with American counterparts, leading to ripple effects in the global fight for reproductive justice.", response="The global implications of the USA Supreme Court ruling on abortion can be significant, as it sets a precedent for other countries and influences the global discourse on reproductive rights. Here are some potential implications:\n\n1. Influence on other countries: The Supreme Court's ruling can serve as a reference point for other countries grappling with their own abortion laws. It can provide legal arguments and reasoning that advocates for reproductive rights can use to challenge restrictive abortion laws in their respective jurisdictions.\n\n2. Strengthening of global reproductive rights movements: A favorable ruling by the Supreme Court can energize and empower reproductive rights movements worldwide. It can serve as a rallying point for activists and organizations advocating for women's rights, leading to increased mobilization and advocacy efforts globally.\n\n3. Counteracting anti-abortion movements: Conversely, a ruling that restricts abortion rights can embolden anti-abortion movements globally. It can provide legitimacy to their arguments and encourage similar restrictive measures in other countries, potentially leading to a rollback of existing reproductive rights.\n\n4. Impact on international aid and policies: The Supreme Court's ruling can influence international aid and policies related to reproductive health. It can shape the priorities and funding decisions of donor countries and organizations, potentially leading to increased support for reproductive rights initiatives or conversely, restrictions on funding for abortion-related services.\n\n5. Shaping international human rights standards: The ruling can contribute to the development of international human rights standards regarding reproductive rights. It can influence the interpretation and application of existing human rights treaties and conventions, potentially strengthening the recognition of reproductive rights as fundamental human rights globally.\n\n6. Global health implications: The Supreme Court's ruling can have implications for global health outcomes, particularly in countries with restrictive abortion laws. It can impact the availability and accessibility of safe and legal abortion services, potentially leading to an increase in unsafe abortions and related health complications.\n\nIt is important to note that the specific implications will depend on the nature of the Supreme Court ruling and the subsequent actions taken by governments, activists, and organizations both within and outside the United States.", reference="The global implications of the USA Supreme Court ruling on abortion are significant. The ruling has led to limited or no access to abortion for one in three women and girls of reproductive age in states where abortion access is restricted. These states also have weaker maternal health support, higher maternal death rates, and higher child poverty rates. Additionally, the ruling has had an impact beyond national borders due to the USA's geopolitical and cultural influence globally. Organizations and activists worldwide are concerned that the ruling may inspire anti-abortion legislative and policy attacks in other countries. The ruling has also hindered progressive law reform and the implementation of abortion guidelines in certain African countries. Furthermore, the ruling has created a chilling effect in international policy spaces, empowering anti-abortion actors to undermine human rights protections.", criteria='is the response correct compared to reference'),
      'output': [AspectCriticOutputWithReference(reason='The response accurately reflects the key points and implications outlined in the reference regarding the global implications of the USA Supreme Court ruling on abortion. It discusses the influence on other countries, the strengthening of global reproductive rights movements, the counteraction of anti-abortion movements, the impact on international aid and policies, the shaping of international human rights standards, and the global health implications, all of which are consistent with the reference provided.', verdict=1)]}}



As you can see, it has the name of the prompt as the key and the input and output as the values. Since, I used AspectCriteriaMetric, the input and output is in the pydantic object used to parse input and output for the metric. You may convert it to a dictionary if needed. For example,


```python
selected_trace = results.traces[0]["answer_correctness"]
selected_trace["0_single_turn_aspect_critic_prompt_with_reference"][
    "input"
].model_dump()
```




    {'user_input': "`user_input`: What are the global implications of the USA Supreme Court ruling on abortion? Answer using `retrieved context`: - In 2022, the USA Supreme Court handed down a decision ruling that overturned 50 years of jurisprudence recognizing a constitutional right to abortion.\n- This decision has had a massive impact: one in three women and girls of reproductive age now live in states where abortion access is either totally or near-totally inaccessible.\n- The states with the most restrictive abortion laws have the weakest maternal health support, higher maternal death rates, and higher child poverty rates.\n- The USA Supreme Court ruling has also had impacts beyond national borders due to the geopolitical and cultural influence wielded by the USA globally and the aid it funds.\n- SRR organizations and activists across the world have expressed fear about the ruling laying the groundwork for anti-abortion legislative and policy attacks in other countries.\n- Advocates have also observed the ruling's impact on progressive law reform and the stalling of the adoption and enforcement of abortion guidelines in certain African countries.\n- The ruling has created a chilling effect in international policy spaces, emboldening anti-abortion state and non-state actors to undermine human rights protections.\nThe USA Supreme Court ruling on abortion has sparked intense debates and discussions not only within the country but also around the world. Many countries look to the United States as a leader in legal and social issues, so the decision could potentially influence the policies and attitudes towards abortion in other nations.\nThe ruling may also impact international organizations and non-governmental groups that work on reproductive rights and women's health issues. Depending on the outcome, there could be shifts in funding, advocacy efforts, and collaborations with American counterparts, leading to ripple effects in the global fight for reproductive justice.",
     'response': "The global implications of the USA Supreme Court ruling on abortion can be significant, as it sets a precedent for other countries and influences the global discourse on reproductive rights. Here are some potential implications:\n\n1. Influence on other countries: The Supreme Court's ruling can serve as a reference point for other countries grappling with their own abortion laws. It can provide legal arguments and reasoning that advocates for reproductive rights can use to challenge restrictive abortion laws in their respective jurisdictions.\n\n2. Strengthening of global reproductive rights movements: A favorable ruling by the Supreme Court can energize and empower reproductive rights movements worldwide. It can serve as a rallying point for activists and organizations advocating for women's rights, leading to increased mobilization and advocacy efforts globally.\n\n3. Counteracting anti-abortion movements: Conversely, a ruling that restricts abortion rights can embolden anti-abortion movements globally. It can provide legitimacy to their arguments and encourage similar restrictive measures in other countries, potentially leading to a rollback of existing reproductive rights.\n\n4. Impact on international aid and policies: The Supreme Court's ruling can influence international aid and policies related to reproductive health. It can shape the priorities and funding decisions of donor countries and organizations, potentially leading to increased support for reproductive rights initiatives or conversely, restrictions on funding for abortion-related services.\n\n5. Shaping international human rights standards: The ruling can contribute to the development of international human rights standards regarding reproductive rights. It can influence the interpretation and application of existing human rights treaties and conventions, potentially strengthening the recognition of reproductive rights as fundamental human rights globally.\n\n6. Global health implications: The Supreme Court's ruling can have implications for global health outcomes, particularly in countries with restrictive abortion laws. It can impact the availability and accessibility of safe and legal abortion services, potentially leading to an increase in unsafe abortions and related health complications.\n\nIt is important to note that the specific implications will depend on the nature of the Supreme Court ruling and the subsequent actions taken by governments, activists, and organizations both within and outside the United States.",
     'reference': "The global implications of the USA Supreme Court ruling on abortion are significant. The ruling has led to limited or no access to abortion for one in three women and girls of reproductive age in states where abortion access is restricted. These states also have weaker maternal health support, higher maternal death rates, and higher child poverty rates. Additionally, the ruling has had an impact beyond national borders due to the USA's geopolitical and cultural influence globally. Organizations and activists worldwide are concerned that the ruling may inspire anti-abortion legislative and policy attacks in other countries. The ruling has also hindered progressive law reform and the implementation of abortion guidelines in certain African countries. Furthermore, the ruling has created a chilling effect in international policy spaces, empowering anti-abortion actors to undermine human rights protections.",
     'criteria': 'is the response correct compared to reference'}



And that's it. Now you have learned how to export and analyze LLM calls made by ragas for evaluation. 
