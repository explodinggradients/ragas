#!/usr/bin/env python3
"""
Text-to-SQL Agent using OpenAI API with comprehensive tracing and logging.

This agent converts natural language queries to SQL queries for database evaluation.
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import dotenv
    dotenv.load_dotenv("../../../.env")
except ImportError:
    # dotenv is optional
    pass

try:
    import openai
except ImportError:
    raise ImportError("openai is required. Install with: pip install openai")

try:
    from .db_utils import execute_sql
except ImportError:
    execute_sql = None


@dataclass
class TraceEvent:
    """Single event in the application trace"""
    event_type: str  # "llm_call", "llm_response", "sql_generation", "error", "init"
    component: str   # "openai_api", "text2sql_agent", "prompt_builder"
    data: Dict[str, Any]


@dataclass
class SQLGenerationResult:
    """Result of SQL generation process"""
    natural_language_query: str
    generated_sql: str
    confidence_score: Optional[float] = None
    generation_time_ms: Optional[float] = None


class Text2SQLAgent:
    """
    Text-to-SQL agent that converts natural language to SQL queries.
    
    Features:
    - Comprehensive trace logging for evaluation
    - Schema-aware query generation
    - Configurable system prompts
    - Export capabilities for Ragas evaluation
    """
    
    def __init__(
        self,
        client,
        model_name: str = "gpt-5-mini",
        prompt_file: Optional[str] = None,
        logdir: str = "text2sql_logs",
    ):
        """
        Initialize the Text-to-SQL agent.
        
        Args:
            client: OpenAI client instance
            model_name: Name of the model to use (default: gpt-5-mini)
            prompt_file: Path to prompt file (default: prompt.txt)
            logdir: Directory to save trace logs
            request_timeout: Per-request timeout (seconds) for OpenAI API
            max_retries: Number of retries for transient OpenAI API errors
        """
        self.client = client
        self.model_name = model_name
        self.traces = []
        self.logdir = Path(logdir)
        # Retry and timeout config
        self.request_timeout = 30
        self.max_retries = 2
        
        # Create log directory if it doesn't exist
        self.logdir.mkdir(exist_ok=True)
        
        # Load prompt
        if prompt_file is None:
            prompt_path = Path(__file__).parent / "prompt.txt"
        else:
            prompt_path = Path(prompt_file)
        
        self.system_prompt = self._load_prompt(prompt_path)
        
        # Initialize trace
        self.traces.append(
            TraceEvent(
                event_type="init",
                component="text2sql_agent",
                data={
                    "model_name": self.model_name,
                    "prompt_file": str(prompt_path),
                    "logdir": str(self.logdir)
                }
            )
        )
    
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
                error_msg = f"Prompt file not found: {prompt_file}"
                self.traces.append(
                    TraceEvent(
                        event_type="error",
                        component="prompt_builder",
                        data={
                            "error_type": "file_not_found",
                            "error_message": error_msg,
                            "prompt_file": str(prompt_file)
                        }
                    )
                )
                raise FileNotFoundError(error_msg)
                
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                error_msg = f"Prompt file is empty: {prompt_file}"
                self.traces.append(
                    TraceEvent(
                        event_type="error",
                        component="prompt_builder",
                        data={
                            "error_type": "empty_file",
                            "error_message": error_msg,
                            "prompt_file": str(prompt_file)
                        }
                    )
                )
                raise IOError(error_msg)
                
            self.traces.append(
                TraceEvent(
                    event_type="prompt_load",
                    component="prompt_builder",
                    data={"prompt_file": str(prompt_file), "success": True}
                )
            )
            return content
                
        except (FileNotFoundError, IOError):
            raise
        except Exception as e:
            error_msg = f"Error reading prompt file {prompt_file}: {e}"
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="prompt_builder",
                    data={
                        "error_type": "read_error",
                        "error_message": error_msg,
                        "prompt_file": str(prompt_file)
                    }
                )
            )
            raise IOError(error_msg) from e
    
    def generate_sql(
        self, 
        natural_query: str, 
        run_id: Optional[str] = None
    ) -> SQLGenerationResult:
        """
        Generate SQL query from natural language input.
        
        Args:
            natural_query: Natural language query to convert
            run_id: Optional run identifier for logging
            
        Returns:
            SQLGenerationResult with generated SQL and metadata
        """
        start_time = datetime.now()
        
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(natural_query) % 10000:04d}"
        
        # Reset traces for new query - start fresh for each generation
        self.traces = []
        
        # Add init trace for this run
        self.traces.append(
            TraceEvent(
                event_type="init",
                component="text2sql_agent",
                data={
                    "run_id": run_id,
                    "model_name": self.model_name,
                    "natural_query": natural_query
                }
            )
        )
        
        logging.info(f"Generating SQL for query: {natural_query} (Run ID: {run_id})")
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": natural_query}
            ]
            
            # Log the API call
            self.traces.append(
                TraceEvent(
                    event_type="llm_call",
                    component="openai_api",
                    data={
                        "model": self.model_name,
                        "natural_query": natural_query,
                        "run_id": run_id,
                        "messages_count": len(messages)
                    }
                )
            )
            
            # Call OpenAI API with simple retries and a per-request timeout
            response = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        timeout=self.request_timeout,
                    )
                    break
                except Exception:
                    # Exponential backoff with small jitter
                    if attempt < self.max_retries:
                        sleep_seconds = min(2 ** attempt, 4) + (0.1 * attempt)
                        time.sleep(sleep_seconds)
                    else:
                        raise
            
            if response is None:
                raise Exception("Failed to get response from OpenAI API after retries")
            
            # Extract generated SQL
            generated_sql = response.choices[0].message.content.strip()
            
            # Clean up the SQL (remove code blocks if present)
            generated_sql = self._clean_sql_output(generated_sql)
            
            end_time = datetime.now()
            generation_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Log the successful response
            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="openai_api",
                    data={
                        "generated_sql": generated_sql,
                        "generation_time_ms": generation_time_ms,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                )
            )
            
            # Create result
            result = SQLGenerationResult(
                natural_language_query=natural_query,
                generated_sql=generated_sql,
                generation_time_ms=generation_time_ms
            )
            
            # Log successful SQL generation
            self.traces.append(
                TraceEvent(
                    event_type="sql_generation",
                    component="text2sql_agent",
                    data={
                        "success": True,
                        "sql_length": len(generated_sql),
                        "sql_preview": generated_sql[:100] + "..." if len(generated_sql) > 100 else generated_sql
                    }
                )
            )
            
            logging.info(f"Successfully generated SQL ({len(generated_sql)} chars)")
            return result
            
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {e}"
            logging.error(error_msg)
            
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={
                        "error_type": "api_error",
                        "error_message": str(e),
                        "natural_query": natural_query
                    }
                )
            )
            
            # Return error result
            return SQLGenerationResult(
                natural_language_query=natural_query,
                generated_sql=f"-- ERROR: {error_msg}",
                generation_time_ms=0.0
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logging.error(error_msg)
            
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="text2sql_agent",
                    data={
                        "error_type": "unexpected_error",
                        "error_message": str(e),
                        "natural_query": natural_query
                    }
                )
            )
            
            # Return error result
            return SQLGenerationResult(
                natural_language_query=natural_query,
                generated_sql=f"-- ERROR: {error_msg}",
                generation_time_ms=0.0
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
        sql_output = re.sub(r'```sql\n?', '', sql_output)
        sql_output = re.sub(r'```\n?', '', sql_output)
        
        # Remove leading/trailing whitespace
        sql_output = sql_output.strip()
        
        # Remove any explanatory text after the SQL (look for common patterns)
        lines = sql_output.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            # Stop at lines that start with explanation markers (but not SQL comments)
            if line.lower().startswith(('this query', 'explanation:', 'note:', 'the query')):
                break
            if line:  # Skip empty lines
                sql_lines.append(line)
        
        return '\n'.join(sql_lines)
    
    def export_traces_to_log(
        self,
        run_id: str,
        natural_query: str,
        result: SQLGenerationResult
    ) -> str:
        """
        Export traces to a log file for evaluation and debugging.
        
        Args:
            run_id: Unique identifier for this run
            natural_query: The natural language query
            result: SQL generation result
            
        Returns:
            Path to the exported log file
        """
        timestamp = datetime.now().isoformat()
        log_filename = f"run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        log_filepath = self.logdir / log_filename
        
        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": self.model_name,
            "natural_query": natural_query,
            "generated_sql": result.generated_sql,
            "generation_time_ms": result.generation_time_ms,
            "traces": [asdict(trace) for trace in self.traces]
        }
        
        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Traces exported to: {log_filepath}")
        return str(log_filepath)


def get_default_agent(
    model_name: str = "gpt-5-mini",
    prompt_file: Optional[str] = None,
    logdir: str = "text2sql_logs",
    request_timeout: int = 60,
    max_retries: int = 2,
) -> Text2SQLAgent:
    """
    Get a default instance of the Text2SQLAgent with OpenAI client.
    
    Args:
        model_name: OpenAI model name
        prompt_file: Path to prompt file
        logdir: Directory for log files
        
    Returns:
        Configured Text2SQLAgent instance
    """
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    agent = Text2SQLAgent(
        client=openai_client,
        model_name=model_name,
        prompt_file=prompt_file,
        logdir=logdir
    )
    # Apply timeout/retry configuration
    agent.request_timeout = request_timeout
    agent.max_retries = max_retries
    return agent


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-SQL Agent")
    parser.add_argument("--query", "-q", type=str, help="Natural language query to convert")
    parser.add_argument("--model", "-m", type=str, default="gpt-5-mini", help="OpenAI model name")
    parser.add_argument("--prompt", "-p", type=str, help="Path to prompt file")
    parser.add_argument("--logdir", "-l", type=str, default="text2sql_logs", help="Log directory")
    parser.add_argument("--test", "-t", action="store_true", help="Run test with sample query")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create agent
    agent = get_default_agent(
        model_name=getattr(args, 'model', 'gpt-5-mini'),
        prompt_file=getattr(args, 'prompt', None),
        logdir=getattr(args, 'logdir', 'text2sql_logs')
    )
    
    if getattr(args, 'test', False):
        # Test with sample query
        test_query = "How much open credit does customer Andrew Bennett?"
        print(f"🧪 Running test with query: {test_query}")
        print("=" * 60)
        
        result = agent.generate_sql(test_query)
        
        print(f"Natural Query: {result.natural_language_query}")
        print(f"Generated SQL: {result.generated_sql}")
        print(f"Generation Time: {result.generation_time_ms:.2f}ms")
        
        # Execute the SQL if db_utils is available
        if execute_sql:
            print("\n" + "=" * 60)
            print("🔍 Executing SQL query against database...")
            
            try:
                success, db_result = execute_sql(result.generated_sql)
                
                if success:
                    print("✅ SQL execution successful!")
                    print("\n📊 Query Results:")
                    try:
                        import pandas as pd
                        if isinstance(db_result, pd.DataFrame):
                            # It's a pandas DataFrame
                            if len(db_result) == 0:
                                print("No rows returned.")
                            else:
                                print(db_result.to_string(index=False))
                                print(f"\nRows returned: {len(db_result)}")
                        else:
                            print(str(db_result))
                    except ImportError:
                        # pandas not available, just print as string
                        print(str(db_result))
                else:
                    print(f"❌ SQL execution failed: {db_result}")
                    
            except Exception as e:
                print(f"❌ Error executing SQL: {e}")
        else:
            print("\n⚠️  db_utils not available - skipping SQL execution")
        
        # Export log
        run_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_file = agent.export_traces_to_log(run_id, test_query, result)
        print(f"\n📝 Log exported to: {log_file}")
            
    elif getattr(args, 'query', None):
        # Single query
        query = getattr(args, 'query', '')
        result = agent.generate_sql(query)
        
        print(f"Natural Query: {result.natural_language_query}")
        print(f"Generated SQL: {result.generated_sql}")
        print(f"Generation Time: {result.generation_time_ms:.2f}ms")
        
        # Export log
        run_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_file = agent.export_traces_to_log(run_id, query, result)
        print(f"Log exported to: {log_file}")
        
    else:
        parser.print_help()
