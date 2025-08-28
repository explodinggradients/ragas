import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def load_prompt(prompt_file: str) -> str:
    """Load prompt from a text file"""
    with open(prompt_file, "r") as f:
        return f.read().strip()

def run_prompt(ticket_text: str, prompt_file: str = "promptv1.txt"):
    """Run the prompt against a customer support ticket"""
    system_prompt = load_prompt(prompt_file)
    user_message = f'Ticket: "{ticket_text}"'
    
    response = client.chat.completions.create(
        model="gpt-5-mini-2025-08-07",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    response = (
        response.choices[0].message.content.strip()
        if response.choices[0].message.content
        else ""
    )
    return response


if __name__ == "__main__":
    # Test with a sample customer support ticket
    test_ticket = "SSO via Okta succeeds then bounces me back to /login with no session. Colleagues can sign in. I tried clearing cookies; same result. Error in devtools: state mismatch. I'm blocked from our boards."
    print("Test ticket:")
    print(f'"{test_ticket}"')
    print("\nResponse:")
    print(run_prompt(test_ticket))
