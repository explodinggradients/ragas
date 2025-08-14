from openai import OpenAI
import os
from .config import BASELINE_MODEL, CANDIDATE_MODEL

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


SYSTEM_PROMPT = """
You are a discount calculation assistant. I will provide a customer profile and you must calculate their discount percentage and explain your reasoning.

Discount rules:
- Age 65+ OR student status: 15% discount
- Annual income < $30,000: 20% discount  
- Premium member for 2+ years: 10% discount
- New customer (< 6 months): 5% discount

Rules can stack up to a maximum of 35% discount.

Respond in JSON format only:
{
  "discount_percentage": number,
  "reason": "clear explanation of which rules apply and calculations",
  "applied_rules": ["list", "of", "applied", "rule", "names"]
}
"""

def run_prompt(prompt: str, model: str = "gpt-4o"):
    """Run the discount calculation prompt with the specified model."""
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
    response = response.choices[0].message.content.strip()
    return response

if __name__ == "__main__":
    customer_profile = """
    Customer Profile:
    - Name: Sarah Johnson
    - Age: 67
    - Student: No
    - Annual Income: $45,000
    - Premium Member: Yes, for 3 years
    - Account Age: 3 years
    """
    print(f"=== System Prompt ===")
    print(SYSTEM_PROMPT)
    print(f"\n=== Customer Profile ===")
    print(customer_profile)
    print(f"\n=== Running Prompts ===")
    print(f"=== Baseline Model ({BASELINE_MODEL}) ===")
    print(run_prompt(customer_profile, model=BASELINE_MODEL))
    
    print(f"\n=== Candidate Model ({CANDIDATE_MODEL}) ===")
    print(run_prompt(customer_profile, model=CANDIDATE_MODEL))
