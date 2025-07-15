"""
Ragas Examples Package

This package contains example implementations and evaluations for various AI/ML use cases
including RAG systems, agents, prompts, and workflows.

Available examples:
- rag: RAG system with document retrieval and response generation
- agent: Mathematical agent with tool calling capabilities  
- prompt: Simple prompt classification example
- workflow: Support ticket triage workflow

Usage:
    python -m ragas_examples <example_name>
    python -m ragas_examples --list
"""

from .rag_eval.rag import ExampleRAG, default_rag_client
from .agent_evals.agent import MathToolsAgent, get_default_agent
from .prompt_evals.prompt import run_prompt
from .workflow_eval.workflow import ConfigurableSupportTriageAgent, default_workflow_client

__all__ = [
    "ExampleRAG",
    "default_rag_client", 
    "MathToolsAgent",
    "get_default_agent",
    "run_prompt",
    "ConfigurableSupportTriageAgent",
    "default_workflow_client",
]