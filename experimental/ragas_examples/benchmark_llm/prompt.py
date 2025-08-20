from openai import OpenAI
import os

DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"

def get_client() -> OpenAI:
    """Lazily create an OpenAI client, requiring the API key only when used.

    This avoids raising errors during module import (e.g., when running --help).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export it before running prompts."
        )
    return OpenAI(api_key=api_key)


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

def run_prompt(prompt: str, model: str = DEFAULT_MODEL):
    """Run the discount calculation prompt with the specified model."""
    client = get_client()
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
    print(f"\n=== Running Prompt with default model {DEFAULT_MODEL} ===")
    print(run_prompt(customer_profile, model=DEFAULT_MODEL))
