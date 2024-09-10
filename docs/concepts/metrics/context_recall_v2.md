# Context Recall

Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out.
In short, recall is about not missing anything important. Since it is about not missing anything, calculating context recall always requires a reference to compare against.

The formula for calculating context recall is as follows:

```{math}
\text{context recall} = {|\text{GT claims that can be attributed to context}| \over |\text{Number of claims in GT}|}
```


## LLM Based Context Recall

Computed using `user_input`, `reference` and the  `retrieved_contexts`, and the values range between 0 and 1, with higher values indicating better performance. This metric uses `reference` as a proxy to `reference_contexts` which also makes it easier to use as annotating reference contexts can be very time consuming. To estimate context recall from the `reference`, the reference is broken down into claims each claim in the `reference` answer is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the reference answer should be attributable to the retrieved context.

## Example
    
```{code-block} python

```

## Non LLM Based Context Recall

Computed using `retrieved_contexts` and `reference_contexts`, and the values range between 0 and 1, with higher values indicating better performance. This metrics uses non llm string comparison metrics to identify if a retrieved context is relevant or not. You can use any non LLM based metrics as distance measure to identify if a retrieved context is relevant or not.

## Example
    
```{code-block} python


```