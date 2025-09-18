"""
Agentic RAG implementation using OpenAI Agents SDK.
The agent can call the BM25 retriever tool multiple times with different queries 
until it gets the right context to answer the user's question.
"""

import asyncio

import mlflow
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

from .data_utils import get_bm25_retriever

load_dotenv(".env")

# Enable auto tracing for OpenAI Agents SDK
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("AgenticRAG")
    mlflow.openai.autolog()
    print("MLflow tracing is enabled.")
except Exception:
    # Tracing is optional; continue without failing if integration is unavailable
    print("MLflow tracing is not available.")
    pass


@function_tool
def bm25_retrieve(query: str) -> str:
    """
    Retrieve relevant documents using BM25 retriever.
    
    Args:
        query: Search query to find relevant documents
    
    Returns:
        String containing the retrieved documents with their content and metadata
    """
    try:
        # Get the BM25 retriever
        retriever = get_bm25_retriever()
        retrieved_docs = retriever.invoke(query)
        
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


def create_agentic_rag_agent() -> Agent:
    """
    Create an agentic RAG agent with BM25 retrieval tool.
    
    Returns:
        Agent instance configured for RAG tasks
    """
    agent = Agent(
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
    return agent


async def query_agentic_rag(question: str) -> str:
    """
    Query the agentic RAG system with a question.
    
    Args:
        question: User's question
        
    Returns:
        Agent's response based on retrieved documents
    """
    agent = create_agentic_rag_agent()
    result = await Runner.run(agent, input=question)
    return result.final_output


async def main():
    """Example usage of the agentic RAG system."""
    
    # Test query
    # question = "What is the default repository type created by the `create_repo` function on Hugging Face Hub?"
    question = "What is the purpose of the `gradio.Blocks` API?"
    
    print(f"Question: {question}")
    print("\n" + "="*50)
    
    try:
        answer = await query_agentic_rag(question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
