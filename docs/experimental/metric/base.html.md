# BaseMetric


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

------------------------------------------------------------------------

<a
href="https://github.com/explodinggradients/ragas_experimental/blob/main/ragas_experimental/metric/base.py#L30"
target="_blank" style="float:right; font-size:smaller">source</a>

### Metric

>  Metric (name:str, prompt:str|ragas_experimental.prompt.base.Prompt,
>              llm:ragas_experimental.llm.llm.RagasLLM)

*Base class for all metrics in the LLM evaluation library.*

### Example

``` python
from ragas_experimental.llm import ragas_llm
from openai import OpenAI

llm = ragas_llm(provider="openai",model="gpt-4o",client=OpenAI())

@dataclass
class CustomMetric(Metric):
    values: t.List[str] = field(default_factory=lambda: ["pass", "fail"])
    
    def _get_response_model(self, with_reasoning: bool) -> t.Type[BaseModel]:
        """Get or create a response model based on reasoning parameter."""
        
        class mymodel(BaseModel):
            result: int
            reason: t.Optional[str] = None
            
        return mymodel 

    def _ensemble(self,results:t.List[MetricResult]) -> MetricResult:
        
        return results[0]  # Placeholder for ensemble logic

my_metric = CustomMetric(name="example", prompt="What is the result of {input}?", llm=llm)
my_metric.score(input="test")
```

    1
