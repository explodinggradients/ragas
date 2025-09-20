"""
Simple RAG implementation using BM25 retriever for document retrieval.
This is used in the evaluate and improve RAG guide.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import mlflow
from dotenv import load_dotenv
from mlflow.entities import SpanType
from openai import AsyncOpenAI

from .retriever import get_bm25_retriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(".env")

# Enable automatic tracing for OpenAI API calls (if available in this MLflow version)
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("SimpleRAG")
    mlflow.openai.autolog()
except Exception:
    # Tracing is optional; continue without failing if integration is unavailable
    pass

class SimpleRAG:
    """
    Simple RAG system that:
    1. Uses BM25 retriever for document retrieval
    2. Uses OpenAI LLM to generate responses based on retrieved documents
    """

    def __init__(
        self,
        openai_client,
        system_prompt: Optional[str] = None,
        model: str = "gpt-5-mini"
    ):
        """
        Initialize RAG system

        Args:
            openai_client: OpenAI client instance
            system_prompt: System prompt template for generation
            model: OpenAI model to use
        """
        self.openai_client = openai_client
        self.model = model
        self.system_prompt = (
            system_prompt
            or """Answer the following question based on the provided documents. 
If the documents don't contain enough information to answer the question, say so clearly.
Be concise in your response.

Question: {query}

Documents:
{context}

Answer:"""
        )

    @mlflow.trace(span_type=SpanType.TOOL)
    def retrieve_documents(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing document info
        """
        # Use the cached BM25 retriever from data_utils
        retriever = get_bm25_retriever()
        retriever.k = top_k
        retrieved_docs = retriever.invoke(query)
        
        # Convert to our format and limit to top_k
        result_docs = []
        for i, doc in enumerate(retrieved_docs[:top_k]):
            result_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "document_id": i
            })
        
        return result_docs

    async def generate_response(self, query: str, top_k: int = 4) -> str:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response using OpenAI
        prompt = self.system_prompt.format(query=query, context=context)

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"

    @mlflow.trace(span_type=SpanType.AGENT)
    async def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary containing response and retrieved documents
        """
        try:
            retrieved_docs = self.retrieve_documents(question, top_k)
            response = await self.generate_response(question, top_k)

            return {
                "answer": response,
                "retrieved_documents": retrieved_docs,
                "num_retrieved": len(retrieved_docs)
            }

        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "retrieved_documents": [],
                "num_retrieved": 0
            }




async def main():
    """Example usage of the Simple RAG system."""
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key: export OPENAI_API_KEY='your_key'"
        )

    openai_client = AsyncOpenAI(api_key=api_key)
    rag_client = SimpleRAG(openai_client=openai_client, model="gpt-5-mini")

    # Test query
    query = "What architecture is the `tokenizers-linux-x64-musl` binary designed for?"
    logger.info(f"Query: {query}")
    
    response = await rag_client.query(query, top_k=3)
    
    logger.info(f"Answer: {response['answer']}")
    logger.info(f"Retrieved {response['num_retrieved']} documents:")
    
    for i, doc in enumerate(response['retrieved_documents'], 1):
        logger.info(f"Document {i}:")
        logger.info(f"Content: {doc['content'][:200]}...")
        if doc['metadata']:
            logger.info(f"Source: {doc['metadata'].get('source', 'Unknown')}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
