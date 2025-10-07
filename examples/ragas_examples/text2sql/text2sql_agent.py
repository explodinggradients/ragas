#!/usr/bin/env python3
"""
Text-to-SQL Agent using OpenAI API.

This agent converts natural language queries to SQL queries for database evaluation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv(".env")

# Configure logger
logger = logging.getLogger(__name__)


class Text2SQLAgent:
    """
    Text-to-SQL agent that converts natural language to SQL queries.

    Features:
    - Schema-aware query generation
    - Configurable system prompts
    """

    def __init__(
        self,
        client,
        model_name: str = "gpt-5-mini",
        prompt_file: Optional[str] = None,
    ):
        """
        Initialize the Text-to-SQL agent.

        Args:
            client: AsyncOpenAI client instance
            model_name: Name of the model to use (default: gpt-5-mini)
            prompt_file: Path to prompt file (default: prompt.txt)
        """
        self.client = client
        self.model_name = model_name

        # Load prompt
        if prompt_file is None:
            prompt_path = Path(__file__).parent / "prompt.txt"
        else:
            prompt_path = Path(prompt_file)

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language input.

        Args:
            question: Natural language query to convert

        Returns:
            Dict with query, sql, and metadata
        """
        logger.info(f"Generating SQL for query: {question}")

        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ]

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            # Extract and clean generated SQL
            generated_sql = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks
            generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

            logger.info(f"Successfully generated SQL ({len(generated_sql)} chars)")
            return {
                "query": question,
                "sql": generated_sql
            }

        except Exception as e:
            error_msg = f"Error: {e}"
            logger.error(error_msg)
            return {
                "query": question,
                "sql": f"-- ERROR: {error_msg}"
            }


# Demo
async def main():
    import os
    from dotenv import load_dotenv
    
    # Load .env from root
    load_dotenv(".env")
    
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Test query
    test_query = "How much open credit does customer Andrew Bennett?"
    
    logger.info("TEXT-TO-SQL AGENT DEMO")
    logger.info("=" * 40)
    
    # Create agent
    logger.info("Creating Text-to-SQL agent...")
    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    agent = Text2SQLAgent(client=openai_client, model_name="gpt-5-mini")
    
    # Generate SQL
    logger.info(f"Query: {test_query}")
    result = await agent.query(test_query)
    
    logger.info(f"Generated SQL: {result['sql']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
