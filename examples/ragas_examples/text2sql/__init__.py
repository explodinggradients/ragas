"""
Text-to-SQL Agent Evaluation Framework

This module provides a comprehensive framework for evaluating Text-to-SQL agents using Ragas.
It includes dataset preparation, agent implementation, evaluation metrics, and error analysis tools.

Key Components:
- Text2SQLAgent: Core agent implementation with OpenAI integration
- Dataset utilities for BookSQL and custom datasets
- Database interface for SQLite query execution
- Ragas-based evaluation framework with custom metrics
- Error analysis and validation tools

Usage:
    from ragas_examples.text2sql import get_default_agent, execute_sql
    
    # Create and use agent
    agent = get_default_agent(model_name="gpt-5-mini")
    result = agent.generate_sql("What is the total revenue?")
    
    # Execute SQL queries
    success, data = execute_sql(result.generated_sql)
"""

from .data_utils import create_sample_dataset, download_booksql_dataset
from .db_utils import SQLiteDB, execute_sql
from .text2sql_agent import Text2SQLAgent, get_default_agent

__all__ = [
    "Text2SQLAgent",
    "get_default_agent", 
    "execute_sql",
    "SQLiteDB",
    "download_booksql_dataset",
    "create_sample_dataset",
]
