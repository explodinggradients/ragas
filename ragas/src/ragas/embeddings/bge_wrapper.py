import typing as t
from sentence_transformers import SentenceTransformer

from ragas.embeddings import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from ragas.cache import CacheInterface


class BGEEmbeddingsWrapper(BaseRagasEmbeddings):
    """
    A wrapper class for BAAI/bge-base-en embeddings within the Ragas framework.
    
    Parameters
    ----------
    model_name : str, optional
        The name of the BGE model to use, by default "BAAI/bge-base-en"
    run_config : RunConfig, optional
        Configuration for the run, by default None
    cache : CacheInterface, optional
        A cache instance for storing results, by default None
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en",
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
    
    def embed_query(self, text: str) -> t.List[float]:
        """Embed a single query text."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Embed multiple documents."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    async def aembed_query(self, text: str) -> t.List[float]:
        """Asynchronously embed a single query text."""
        # For sentence-transformers, we'll use the synchronous version
        # since it's already optimized for batch processing
        return self.embed_query(text)
    
    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Asynchronously embed multiple documents."""
        # For sentence-transformers, we'll use the synchronous version
        # since it's already optimized for batch processing
        return self.embed_documents(texts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
