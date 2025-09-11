from pathlib import Path

from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv("../../../.env")


def load_vectorstore_from_disk(vectorstore_path="vector_store"):
    """Load FAISS vectorstore from disk."""
    vectorstore_dir = Path(vectorstore_path)
    
    # Check if vectorstore exists
    if not vectorstore_dir.exists() or not (vectorstore_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"Vectorstore not found at {vectorstore_path}. "
            "Please run 'python data_preparation.py' first to create the vectorstore."
        )
    
    # Load embeddings (same as used during creation)
    embeddings = OpenAIEmbeddings()
    
    # Load vectorstore from disk
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"✓ Loaded vectorstore from {vectorstore_path}")
    return vectorstore


def create_retriever_tool_from_vectorstore(vectorstore):
    """Create a retriever tool from a loaded vectorstore."""
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_ragas_docs",
        "Search and return information about Ragas documentation including concepts, metrics, datasets, and experimentation.",
    )
    
    print("✓ Created retriever tool")
    return retriever_tool


def main():
    """Main function to load vectorstore and create retriever tool."""
    try:
        # Load vectorstore from disk
        print("=== Loading RAG System ===")
        vectorstore = load_vectorstore_from_disk()
        
        # Create retriever tool
        retriever_tool = create_retriever_tool_from_vectorstore(vectorstore)
        
        # Test the tool
        print("\n--- Testing retriever tool ---")
        result = retriever_tool.invoke({"query": "What are the different types of metrics in Ragas?"})
        print(f"Query result: {result}")
        
        return retriever_tool
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 To fix this:")
        print("1. Run: python data_preparation.py")
        print("2. Then run: python rag.py")
        return None


if __name__ == "__main__":
    main()
