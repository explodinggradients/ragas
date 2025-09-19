"""
Agentic RAG implementation using OpenAI Agents SDK.
The agent can call the BM25 retriever tool multiple times with different queries 
until it gets the right context to answer the user's question.
"""

import asyncio
import logging

import mlflow
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

from .data_utils import get_bm25_retriever

load_dotenv(".env")

# Setup logger
logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Agentic RAG system that uses an AI agent with BM25 retrieval capabilities.
    
    The agent can strategically call the retriever multiple times with different
    queries to gather comprehensive context before answering questions.
    """
    
    def __init__(self, enable_mlflow: bool = True):
        """
        Initialize the Agentic RAG system.
        
        Args:
            enable_mlflow: Whether to enable MLflow tracing
        """
        self.retriever = None
        self.agent = None
        self.mlflow_enabled = False
        
        if enable_mlflow:
            self._setup_mlflow()
        
        self._setup_agent()
    
    def _setup_mlflow(self):
        """Setup MLflow tracing for the agents."""
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("AgenticRAG")
            mlflow.openai.autolog()
            self.mlflow_enabled = True
            logger.info("MLflow tracing is enabled.")
        except Exception:
            # Tracing is optional; continue without failing if integration is unavailable
            logger.warning("MLflow tracing is not available.")
            self.mlflow_enabled = False
    
    def _bm25_retrieve_tool(self, query: str) -> str:
        """
        Internal method to create the BM25 retrieval tool for the agent.
        
        Args:
            query: Search query to find relevant documents
        
        Returns:
            String containing the retrieved documents with their content and metadata
        """
        try:
            # Get the BM25 retriever lazily
            if self.retriever is None:
                self.retriever = get_bm25_retriever()
            
            retrieved_docs = self.retriever.invoke(query)
            
            if not retrieved_docs:
                return "No relevant documents found for this query."
            
            # Format the documents for the agent
            result_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                doc_info = f"Document {i}:\nContent: {doc.page_content}"
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown')
                    doc_info += f"\nSource: {source}"
                result_parts.append(doc_info)
            
            return "\n\n".join(result_parts)
            
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"
    
    def _setup_agent(self):
        """Setup the agentic RAG agent with retrieval capabilities."""
        
        # Create the function tool from the instance method
        @function_tool
        def bm25_retrieve(query: str) -> str:
            """
            Retrieve relevant documents using BM25 retriever.
            
            Args:
                query: Search query to find relevant documents
            
            Returns:
                String containing the retrieved documents with their content and metadata
            """
            return self._bm25_retrieve_tool(query)
        
        self.agent = Agent(
            name="Agentic RAG Assistant",
            instructions="""You are a helpful RAG assistant that can search through documents to answer questions.

You have access to a BM25 document retriever tool. Use this tool strategically:

SEARCH STRATEGY FOR BM25:
- Use SHORT, SPECIFIC keyword queries (e.g., "gradio Blocks" not "purpose of gradio Blocks API")
- Extract key technical terms from the question and search for those directly
- Try different combinations of keywords if first search doesn't work
- ALWAYS do multiple searches with different keyword combinations
- If you get irrelevant results, try more specific or alternative keywords

SEARCH BEHAVIOR:
- MANDATORY: Always do at least 2-3 tool calls with different search strategies
- First search: Use main keywords from the question
- Second search: Try alternative keywords or more specific terms
- Third search (if needed): Try broader or related terms
- Don't give up after one search - keep trying different keyword combinations

ANSWERING:
- Only provide an answer when you have relevant context from the retrieved documents
- If multiple searches return irrelevant results, clearly state the documents don't contain the needed information
- Always cite the documents you use in your answer

Remember: BM25 works best with specific keywords, not full questions or verbose phrases!""",
            tools=[bm25_retrieve],
        )
    
    async def query(self, question: str) -> str:
        """
        Query the agentic RAG system with a question.
        
        Args:
            question: User's question
            
        Returns:
            Agent's response based on retrieved documents
        """
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call _setup_agent() first.")
        
        result = await Runner.run(self.agent, input=question)
        return result.final_output


async def main():
    """Example usage of the agentic RAG system."""
    
    # Test query
    # question = "What is the default repository type created by the `create_repo` function on Hugging Face Hub?"
    question = "What is the purpose of the `gradio.Blocks` API?"
    
    logger.info(f"Question: {question}")
    logger.info("\n" + "="*50)
    
    try:
        # Use the new class-based approach
        rag_system = AgenticRAG(enable_mlflow=True)
        answer = await rag_system.query(question)
        logger.info(f"Answer: {answer}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
