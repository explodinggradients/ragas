from llama_index import (
    GPTVectorStoreIndex,
    ResponseSynthesizer,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# openai embeddings
openai_sc = ServiceContext.from_defaults()

# load index
index = load_index_from_storage(storage_context)


def query(prompt: str, k: int = 3) -> str:
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=k,
    )

    # configure response synthesizer
    response_synthesizer = ResponseSynthesizer.from_args(
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
    )

    # assemble query engine
    qe = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    r = qe.query(prompt)
    if r is None:
        return "Sorry, I don't know the answer to that."

    return r.response


def get_answers_and_context(prompt: str) -> tuple[str, str, str, list]:
    r = query(prompt)
    c = [sn.node.text for sn in r.source_nodes]
    return (r.response, query(prompt, k=1), "nothing", c)
