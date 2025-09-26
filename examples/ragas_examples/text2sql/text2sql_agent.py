#!/usr/bin/env python3
"""
Text-to-SQL Agent using OpenAI API.

This agent converts natural language queries to SQL queries for database evaluation.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import dotenv
import openai

dotenv.load_dotenv("../../../.env")

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class SQLGenerationResult:
    """Result of SQL generation process"""

    natural_language_query: str
    generated_sql: str


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
            client: OpenAI client instance
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

        self.system_prompt = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_file: Path) -> str:
        """
        Load system prompt from file.

        Args:
            prompt_file: Path to the prompt file

        Returns:
            System prompt string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
            IOError: If there's an error reading the file
        """
        try:
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                raise IOError(f"Prompt file is empty: {prompt_file}")

            logger.info(f"Loaded prompt from: {prompt_file}")
            return content

        except Exception as e:
            error_msg = f"Error reading prompt file {prompt_file}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    def generate_sql(self, natural_query: str) -> SQLGenerationResult:
        """
        Generate SQL query from natural language input.

        Args:
            natural_query: Natural language query to convert

        Returns:
            SQLGenerationResult with generated SQL
        """
        logger.info(f"Generating SQL for query: {natural_query}")

        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": natural_query},
            ]

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            # Extract generated SQL
            generated_sql = response.choices[0].message.content.strip()

            # Clean up the SQL (remove code blocks if present)
            generated_sql = self._clean_sql_output(generated_sql)

            # Create result
            result = SQLGenerationResult(
                natural_language_query=natural_query, generated_sql=generated_sql
            )

            logger.info(f"Successfully generated SQL ({len(generated_sql)} chars)")
            return result

        except openai.APIError as e:
            error_msg = f"OpenAI API error: {e}"
            logger.error(error_msg)

            # Return error result
            return SQLGenerationResult(
                natural_language_query=natural_query,
                generated_sql=f"-- ERROR: {error_msg}",
            )

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)

            # Return error result
            return SQLGenerationResult(
                natural_language_query=natural_query,
                generated_sql=f"-- ERROR: {error_msg}",
            )

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


def get_default_agent(
    model_name: str = "gpt-5-mini",
    prompt_file: Optional[str] = None,
) -> Text2SQLAgent:
    """
    Get a default instance of the Text2SQLAgent with OpenAI client.

    Args:
        model_name: OpenAI model name
        prompt_file: Path to prompt file

    Returns:
        Configured Text2SQLAgent instance
    """
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return Text2SQLAgent(
        client=openai_client, model_name=model_name, prompt_file=prompt_file
    )


# Demo
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL Agent")
    parser.add_argument(
        "--query", "-q", type=str, help="Natural language query to convert"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-5-mini", help="OpenAI model name"
    )
    parser.add_argument("--prompt", "-p", type=str, help="Path to prompt file")
    parser.add_argument(
        "--test", "-t", action="store_true", help="Run test with sample query"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create agent
    agent = get_default_agent(model_name=args.model, prompt_file=args.prompt)

    if args.test:
        # Test with sample query
        test_query = "How much open credit does customer Andrew Bennett?"
        logger.info(f"Running test with query: {test_query}")
        logger.info("=" * 60)

        result = agent.generate_sql(test_query)

        print(f"Natural Query: {result.natural_language_query}")
        print(f"Generated SQL: {result.generated_sql}")

    elif args.query:
        # Single query
        result = agent.generate_sql(args.query)

        print(f"Natural Query: {result.natural_language_query}")
        print(f"Generated SQL: {result.generated_sql}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
