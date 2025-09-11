from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv("../../../.env")


def download_data():
    """Download raw markdown files from GitHub to a documents folder."""
    files = {
        "experimentation.md": "docs/concepts/experimentation.md",
        "index.md": "docs/concepts/metrics/overview/index.md", 
        "datasets.md": "docs/concepts/datasets.md"
    }
    
    base_url = "https://raw.githubusercontent.com/explodinggradients/ragas/main/"
    docs_folder = Path("documents")
    docs_folder.mkdir(exist_ok=True)
    
    for filename, path in files.items():
        try:
            response = requests.get(f"{base_url}{path}")
            response.raise_for_status()
            (docs_folder / filename).write_text(response.text, encoding='utf-8')
            print(f"✓ {filename}")
        except Exception as e:
            print(f"✗ {filename}: {e}")


def load_and_split_documents():
    """Load documents from directory and split them using recursive text splitter."""
    # Load documents using DirectoryLoader
    path = "documents/"  # Using the documents folder created by download_data()
    loader = DirectoryLoader(path, glob="**/*.md")
    docs = loader.load()
    
    # Create recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, 
        chunk_overlap=50
    )
    
    # Split documents
    doc_splits = text_splitter.split_documents(docs)
    
    print(f"✓ Loaded {len(docs)} documents")
    print(f"✓ Split into {len(doc_splits)} chunks")
    
    return doc_splits


def create_and_save_vectorstore(doc_splits, vectorstore_path="vector_store"):
    """Create FAISS vectorstore from document splits and save to disk."""
    # Create vector store with OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(
        documents=doc_splits, 
        embedding=embeddings
    )
    
    # Create directory if it doesn't exist
    Path(vectorstore_path).mkdir(exist_ok=True)
    
    # Save vectorstore to disk
    vectorstore.save_local(vectorstore_path)
    
    print(f"✓ Created and saved FAISS vectorstore to {vectorstore_path}")
    return vectorstore


def vectorstore_exists(vectorstore_path="vector_store"):
    """Check if vectorstore already exists on disk."""
    vectorstore_dir = Path(vectorstore_path)
    # Check for both index.faiss and index.pkl files
    return (vectorstore_dir.exists() and 
            (vectorstore_dir / "index.faiss").exists() and 
            (vectorstore_dir / "index.pkl").exists())


def main():
    """Main function to prepare and save the vectorstore."""
    vectorstore_path = "vector_store"
    
    # Check if vectorstore already exists
    if vectorstore_exists(vectorstore_path):
        print(f"✓ Vectorstore already exists at {vectorstore_path}")
        print("To regenerate, delete the vector_store directory and run again.")
        return
    
    print("=== Starting Data Preparation ===")
    
    # Download data
    print("\n1. Downloading documents...")
    download_data()
    
    # Load and split documents
    print("\n2. Loading and splitting documents...")
    doc_splits = load_and_split_documents()
    
    # Create and save vectorstore
    print("\n3. Creating and saving vectorstore...")
    create_and_save_vectorstore(doc_splits, vectorstore_path)
    
    print("\n=== Data Preparation Complete ===")
    print(f"Vectorstore saved to: {vectorstore_path}")
    print("You can now run rag.py to use the pre-built vectorstore.")


if __name__ == "__main__":
    main()
