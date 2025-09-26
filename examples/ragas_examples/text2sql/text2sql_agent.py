#!/usr/bin/env python3
"""
Text-to-SQL Agent using OpenAI API.

This agent converts natural language queries to SQL queries for database evaluation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import dotenv
from openai import AsyncOpenAI
import openai

dotenv.load_dotenv("../../../.env")

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

            # Extract generated SQL
            generated_sql = response.choices[0].message.content.strip()

            # Clean up the SQL (remove code blocks if present)
            generated_sql = self._clean_sql_output(generated_sql)

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

    def _clean_sql_output(self, sql_output: str) -> str:
        """
        Clean the SQL output from the LLM.

        Args:
            sql_output: Raw SQL output from the model

        Returns:
            Cleaned SQL query
        """
        # Remove markdown code blocks
        sql_output = re.sub(r"```sql\n?", "", sql_output)
        sql_output = re.sub(r"```\n?", "", sql_output)

        # Remove leading/trailing whitespace
        sql_output = sql_output.strip()

        # Remove any explanatory text after the SQL (look for common patterns)
        lines = sql_output.split("\n")
        sql_lines = []

        for line in lines:
            line = line.strip()
            # Stop at lines that start with explanation markers (but not SQL comments)
            if line.lower().startswith(
                ("this query", "explanation:", "note:", "the query")
            ):
                break
            if line:  # Skip empty lines
                sql_lines.append(line)

        return "\n".join(sql_lines)




# Demo
async def main():
    import os
    import pathlib
    from dotenv import load_dotenv
    
    # Load .env from root
    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    load_dotenv(root_dir / ".env")
    
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
