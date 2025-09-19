from functools import lru_cache
from threading import Lock
from typing import Any, Dict, Iterable, cast

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from tqdm import tqdm

import datasets

_retriever_build_lock = Lock()

@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """
    Build and cache a BM25 retriever on first use.
    This function is thread-safe and will only construct the retriever once
    per Python process. Subsequent calls return the cached instance.
    """
    with _retriever_build_lock:
        print("Loading dataset for BM25 retriever...")
        knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
        kb_iter: Iterable[Dict[str, Any]] = cast(
            Iterable[Dict[str, Any]], knowledge_base
        )

        source_documents = [
            Document(
                page_content=row["text"],
                metadata={"source": row["source"].split("/")[1]},
            )
            for row in kb_iter
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        # Split docs and keep only unique ones
        print("Splitting documents for BM25 retriever...")
        processed_chunks = []
        seen_texts = {}
        for document in tqdm(source_documents):
            new_docs = text_splitter.split_documents([document])
            for new_doc in new_docs:
                if new_doc.page_content not in seen_texts:
                    seen_texts[new_doc.page_content] = True
                    processed_chunks.append(new_doc)

        print("Creating BM25 retriever...")
        retriever = BM25Retriever.from_documents(
            documents=processed_chunks,
            k=4,  # default; callers can override via retriever.k
        )
        return retriever
