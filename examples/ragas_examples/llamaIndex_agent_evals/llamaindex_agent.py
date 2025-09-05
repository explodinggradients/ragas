import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.google_genai import GoogleGenAI


# Define tools to manage our shopping list
async def add_item(ctx: Context, item: str) -> str:
    """Add an item to the shopping list and return confirmation."""
    async with ctx.store.edit_state() as ctx_state:
        if item.lower() not in [i.lower() for i in ctx_state["state"]["shopping_list"]]:
            ctx_state["state"]["shopping_list"].append(item)
            return f"Added '{item}' to the shopping list"
        else:
            return f"'{item}' is already in the shopping list"


async def remove_item(ctx: Context, item: str) -> str:
    """Remove an item from the shopping list by name."""
    async with ctx.store.edit_state() as ctx_state:
        for i, list_item in enumerate(ctx_state["state"]["shopping_list"]):
            if list_item.lower() == item.lower():
                ctx_state["state"]["shopping_list"].pop(i)
                return f"Removed '{list_item}' from the shopping list"

        return f"'{item}' was not found in the shopping list"


async def list_items(
    ctx: Context,
) -> str:
    """List all items in the shopping list."""
    async with ctx.store.edit_state() as ctx_state:
        shopping_list = ctx_state["state"]["shopping_list"]

        if not shopping_list:
            return "The shopping list is empty."

        items_text = "\n".join([f"- {item}" for item in shopping_list])
        return f"Current shopping list:\n{items_text}"


llm = GoogleGenAI(model="gemini-2.0-flash", api_key=os.environ["GOOGLE_API_KEY"])

workflow = FunctionAgent(
    tools=[add_item, remove_item, list_items],
    llm=llm,
    system_prompt="""Your job is to manage a shopping list.
The shopping list starts empty. You can add items, remove items by name, and list all items.""",
    initial_state={"shopping_list": []},
)
