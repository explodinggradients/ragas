"""RAG Evaluation Example

This module contains a complete RAG (Retrieval-Augmented Generation) system
with document retrieval, response generation, and evaluation capabilities.
"""

from .rag import ExampleRAG, default_rag_client
from .evals import run_experiment, load_dataset

__all__ = ["ExampleRAG", "default_rag_client", "run_experiment", "load_dataset"]