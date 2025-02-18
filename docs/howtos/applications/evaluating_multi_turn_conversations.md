# Evaluating Multi-Turn Conversations

This tutorial is inspired by Hamel’s notes on evaluating multi-turn conversations for LLM-based applications. The goal is to create a simple and actionable evaluation framework using Ragas metrics that clearly defines what makes a conversation successful. By the end of this tutorial, you will be able to perform multi-turn evaluations based on insights gathered from the error analysis of your AI application.

### Ragas Metrics  

Ragas offers **AspectCritic**, a powerful evaluation metric for assessing multi-turn conversations with binary outcomes. It helps determine whether a conversation meets predefined success criteria.  

**[AspectCritic](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/general_purpose/#aspect-critic)**  
AspectCritic evaluates responses based on predefined aspects written in free-form natural language. It returns a binary output indicating whether the response aligns with the defined aspect.

This metric aligns with Hamel's [suggestion](https://hamel.dev/notes/llm/officehours/evalmultiturn.html#focus-on-binary-decisions) to focus on binary decisions, which eliminate ambiguity and provide a clear, actionable approach to improving conversation quality.

### Practical Example – Evaluating a Banking Chatbot  

When evaluating, focus on metrics that directly align with your users’ needs. Any change in the score should reflect a meaningful impact on the user experience.  

Let’s consider an example where you are building a chatbot for a bank.  

After conducting [error analysis](https://hamel.dev/notes/llm/officehours/erroranalysis.html#the-data-first-approach), you find that the chatbot occasionally forgets tasks it was asked to complete or performs them only partially. To improve the chatbot’s performance, you need a reliable method to **measure and evaluate** this behavior.  

> **Note:** When defining the scoring criteria, use standardized terminology.  
> - Refer to the user’s message as `human` message.  
> - Refer to the chatbot’s message as `AI` message.  


```python
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

load_dotenv()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
```


```python
from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate


sample1 = MultiTurnSample(
    user_input=[
        HumanMessage(
            content="I need to increase my credit limit and check why my last transaction at Walmart was declined."
        ),
        AIMessage(
            content="Let's check the declined transaction first, the $234.56 transaction at Walmart on November 20th was declined due to insufficient funds."
        ),
        HumanMessage(content="That's not possible, I had enough money in my account."),
        AIMessage(
            content="I understand your concern. Let me check your account details... I see that while your current balance is $1,500, there was a pending hotel reservation hold of $800 at the time of the Walmart transaction, which reduced your available balance to $700."
        ),
        HumanMessage(content="Oh, I forgot about the hotel booking."),
        AIMessage(content="Is there anything else I can assist you with?"),
        HumanMessage(content="What about increasing my credit limit?"),
        AIMessage(
            content="Based on your account history and current credit score of 740, you're eligible for a credit limit increase. The maximum increase available is $2,000. However, I notice you had three declined transactions this month, so I'd recommend reviewing your spending patterns before accepting an increase."
        ),
    ]
)

sample2 = MultiTurnSample(
    user_input=[
        HumanMessage(
            content="I need to check why my last transaction at Target was declined."
        ),
        AIMessage(
            content="Let's look at the declined transaction. The $145.30 transaction at Target on January 5th was declined due to a suspected fraud alert."
        ),
        HumanMessage(content="That’s strange, I didn’t report anything suspicious."),
        AIMessage(
            content="I understand. Let me look deeper. It seems a fraud prevention team flagged your recent purchase at a different store for unusually high activity, which triggered the decline at Target as well."
        ),
        HumanMessage(content="Ah, that makes sense. I did shop a lot that day."),
        AIMessage(content="Is there anything else I can assist you with?"),
    ]
)
```


```python
definition = "Return 1 if the AI completes all Human requests fully without any rerequests; otherwise, return 0."

aspect_critic = AspectCritic(
    name="forgetfulness_aspect_critic",
    definition=definition,
    llm=evaluator_llm,
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample1, sample2]),
    metrics=[aspect_critic],
)

result.to_pandas()
```
Output
```
Evaluating: 100%|██████████| 2/2 [00:00<?, ?it/s]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>forgetfulness_aspect_critic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'I need to increase my credit lim...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': 'I need to check why my last tran...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



When evaluating with LLM-based metrics, each metric may involve one or more calls to the LLM. The traces of evaluation can provide insghts for understanding the results and diagnosing any issues. You can find more details on this process by visiting [this page](https://docs.ragas.io/en/stable/howtos/applications/_metrics_llm_calls/).

Another pattern identified during error analysis is that your banking chatbot occasionally drifts from discussing basic account services into offering unauthorized investment advice. To maintain user trust and ensure regulatory compliance, you want the system to implement **graceful transitions** when conversations approach these boundaries. You can achieve this by defining a metric like the one below.


```python
definition = "Return 1 if the AI stays within the banking domain, providing services like account-related information, credit card details, or general banking advice, and avoids offering financial or investment advice; otherwise, return 0"

aspect_critic = AspectCritic(
    name="Banking Compliance Metric(aspect critic)",
    definition=definition,
    llm=evaluator_llm,
)
```


```python
sample1 = MultiTurnSample(
    user_input=[
        HumanMessage(
            content="I need to increase my credit limit and check why my last transaction at Walmart was declined."
        ),
        AIMessage(
            content="Let's check the declined transaction first, the $234.56 transaction at Walmart on November 20th was declined due to insufficient funds."
        ),
        HumanMessage(content="That's not possible, I had enough money in my account."),
        AIMessage(
            content="I understand your concern. Let me check your account details... I see that while your current balance is $1,500, there was a pending hotel reservation hold of $800 at the time of the Walmart transaction, which reduced your available balance to $700."
        ),
        HumanMessage(content="Oh, I forgot about the hotel booking."),
        AIMessage(content="Is there anything else I can assist you with?"),
        HumanMessage(content="What about increasing my credit limit?"),
        AIMessage(
            content="Based on your account history and current credit score of 740, you're eligible for a credit limit increase. The maximum increase available is $2,000. However, I notice you had three declined transactions this month, so I'd recommend reviewing your spending patterns before accepting an increase."
        ),
        HumanMessage(
            content="What investment options do you recommend with my spare money?"
        ),
        AIMessage(
            content="While I can help you with banking transactions and credit card matters, I'm not authorized to provide investment advice. I'd be happy to connect you with our certified financial advisor who can help you explore investment options."
        ),
    ]
)

sample2 = MultiTurnSample(
    user_input=[
        HumanMessage(content="Hi there!"),
        AIMessage(content="Hello! How can I help you today?"),
        HumanMessage(content="How much money do I have in my bank account?"),
        AIMessage(
            content="Your current balance is ₹85,750. Let me know if you’d like details on recent transactions or account services!"
        ),
        HumanMessage(
            content="Which are better financial instruments for wealth generation—crypto like Bitcoin or stocks?"
        ),
        AIMessage(
            content="Both stocks and cryptocurrencies have the potential for wealth generation, but they come with different risk levels. Stocks are usually more stable, linked to company performance, and can offer long-term growth. Cryptocurrencies, like Bitcoin, are highly volatile but can deliver quick, high returns. Diversification might be a smart approach based on your risk tolerance and goals."
        ),
        HumanMessage(content="Interesting! Thanks for the advice."),
    ]
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample1, sample2]),
    metrics=[aspect_critic],
)

result.to_pandas()
```
Output
```
Evaluating: 100%|██████████| 2/2 [00:00<?, ?it/s]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>Banking Compliance Metric(aspect critic)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'I need to increase my credit lim...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': 'Hi there!', 'type': 'human'}, {'...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Tonality

In this section, we will explore how to evaluate whether your chatbot maintains a consistent tone across different regions and cultures — one of the most challenging aspects of multilingual deployment.

What is considered polite in one culture may be perceived differently in another. For example, politeness in Japan is often expressed with a formal, indirect, and respectful tone, while in Mexico, it is typically conveyed with a warm, friendly, and engaging tone.

To ensure your chatbot adapts to these cultural nuances, you can define custom evaluation metrics that assess whether the tone aligns with the expectations of each target audience.


```python
japanese_polite_definition = "Return 1 if the AI maintains a formal, polite, and respectful tone throughout the conversation, using indirect expressions and avoiding overly casual or direct language; otherwise, return 0."

mexican_polite_definition = "Return 1 if the AI maintains a warm, friendly, and engaging tone, blending informality with respect and fostering a personable, approachable interaction; otherwise, return 0."

jpn_polite_aspect_critic = AspectCritic(
    name="japanese_polite_aspect_critic",
    definition=japanese_polite_definition,
    llm=evaluator_llm,
)

mex_polite_aspect_critic = AspectCritic(
    name="mexican_polite_aspect_critic",
    definition=mexican_polite_definition,
    llm=evaluator_llm,
)
```


```python
sample1 = MultiTurnSample(
    user_input=[
        HumanMessage(content="I forgot my PIN. What should I do?"),
        AIMessage(
            content="I can assist you in resetting your PIN. May I send a verification code to your registered phone number?"
        ),
        HumanMessage(content="Yes, please."),
        AIMessage(content="I have sent the code. Please check and enter it here."),
        HumanMessage(content="I have entered it."),
        AIMessage(content="Thank you. Please set a new PIN."),
        HumanMessage(content="I have set it."),
        AIMessage(
            content="Your PIN reset is complete. Thank you for choosing our bank."
        ),
    ]
)

sample2 = MultiTurnSample(
    user_input=[
        HumanMessage(content="Help! I forgot my PIN."),
        AIMessage(
            content="Don't worry! I’ll help you reset it. Do you want me to send a verification code to your registered number?"
        ),
        HumanMessage(content="Yes, please."),
        AIMessage(content="Done! Check your phone and enter the code here. "),
        HumanMessage(content="I have entered it."),
        AIMessage(content="Great! Now choose a new PIN and keep it in a safe place."),
        HumanMessage(content="Done."),
        AIMessage(content="Perfect! All set. If you need more help, I’ll be here."),
    ]
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample1, sample2]),
    metrics=[jpn_polite_aspect_critic, mex_polite_aspect_critic],
)

result.to_pandas()
```
Output
```
Evaluating: 100%|██████████| 4/4 [00:00<?, ?it/s]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>japanese_polite_aspect_critic</th>
      <th>mexican_polite_aspect_critic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'I forgot my PIN. What should I d...</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': 'Help! I forgot my PIN.', 'type':...</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The above evaluation result highlights that what is considered polite in Mexico may not be perceived as polite in Japan.

### Checking for Brand Tone  

In this section, we will explore how to evaluate whether the chatbot’s tone is consistent with the business’s values, target audience, and overall brand identity.  

**What is a Brand Tone of Voice?**
A brand’s tone of voice refers to its choice of words when communicating with its audience in written or spoken interactions. By defining a unique tone of voice, brands can develop an authentic personality, style, and attitude.  
[Reference](https://filestage.io/blog/brand-tone-of-voice-examples/)  

For example:  

**Google – Informative and Helpful Brand Voice**  
Have you noticed how simple and intuitive everything feels when you use Google products? But as soon as you switch to another tool, things suddenly feel more complicated. This seamless experience results from Google’s mastery of its brand voice.  

Google maintains a friendly and approachable tone while keeping user communication clear and concise. Their entire brand voice revolves around being helpful, clear, and accessible, making their products intuitive for everyone.  
[Reference](https://filestage.io/blog/brand-tone-of-voice-examples/)  

You can assess whether your chatbot’s responses align with your brand identity by defining a custom evaluation metric like the one below.


```python
definition = "Return 1 if the AI's communication is friendly, approachable, helpful, clear, and concise; otherwise, return 0."

aspect_critic = AspectCritic(
    name="Brand Voice Metric(aspect critic)",
    definition=definition,
    llm=evaluator_llm,
)
```


```python
sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="Hi! I want to apply for a credit card."),
        AIMessage(
            content="Sure! We’ve got a few great options. What kind of benefits are you interested in?"
        ),
        HumanMessage(content="Travel rewards, mostly."),
        AIMessage(
            content="Perfect! Our Travel Rewards Card gives you miles for every dollar spent on flights and hotels. Want to apply now?"
        ),
        HumanMessage(content="Yes, please."),
        AIMessage(
            content="Awesome! Here’s the link to apply. It’s quick and easy. Let me know if you need help!"
        ),
    ]
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample]),
    metrics=[aspect_critic],
)

result.to_pandas()
```
Output
```
Evaluating:   100%|██████████| 1/1 [00:00<?, ?it/s]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>Brand Voice Metric(aspect critic)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'Hi! I want to apply for a credit...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


