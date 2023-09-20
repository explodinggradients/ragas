from llama_index import OpenAIEmbedding
from numpy.random import default_rng
import numpy as np
from typing import t
from llama_index.readers.schema.base import Document
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.schema import TextNode
from langchain.chat_models import ChatOpenAI

from ragas.testset.prompts import COMPRESS_QUESTION, CONDITIONAL_QUESTION, CONVERSATION_QUESTION, FILTER_QUESTION, MULTICONTEXT_QUESTION, REASONING_QUESTION, SCORE_CONTEXT, SEED_QUESTION
from ragas.metrics.llms import generate
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate


DEFAULT_TESTDISTRIBUTION = {
    "simple": 0.5,
    "reasoning": 0.2,
    "multi_context": 0.2,
    "conditional": 0.1
}

question_deep_map = {
    "reasoning": "_reasoning_question",
    "conditional": "_condition_question"
}

class TestsetGenerator:
    
    """
    Ragas Test Set Generator
    
    Attributes
    ----------
    test_distribution : dict
        Distribution of different types of questions to be generated from given
        set of documents. Defaults to {"easy":0.1, "reasoning":0.4, "conversation":0.5}
    """
    
    def __init__(self, generator_llm: BaseLLM | BaseChatModel, 
                 filter_llm: BaseLLM | BaseChatModel, 
                 embedding_model: Embeddings,
                 testset_distribution: t.Optional[t.Dict[str, float]] = None,
                 chat_qa: bool = True, chunk_size: int = 1024, seed: int = 42) -> None:
        
        self.generator_llm = generator_llm
        self.filter_llm = filter_llm
        self.embedding_model = embedding_model
        assert sum(testset_distribution.values()) == 1, "Sum of distribution should be 1"
        testset_distribution = testset_distribution or DEFAULT_TESTDISTRIBUTION
        
        probs = np.cumsum(testset_distribution.values())
        types = testset_distribution.keys()
        self.testset_distribution = dict(zip(types, probs))
        
        self.chat_qa = chat_qa
        self.chunk_size = chunk_size
        self.threshold = 7.5
        self.rng = default_rng(seed)
        
    @classmethod
    def from_default(cls, openai_generator_llm: str = "gpt-3.5-turbo-16k",
                     openai_filter_llm: str = "gpt-4", embeddings_model: Embeddings | None = None):
        
        generator_llm = ChatOpenAI(model_name = openai_generator_llm)
        filtering_llm = ChatOpenAI(model_name = openai_filter_llm)
        embeddings_model = embeddings_model or OpenAIEmbedding()
        return cls(generator_llm = generator_llm,
                   filtering_llm = filtering_llm,
                   embeddings_model = embeddings_model)
        
    def _get_evolve_type(self):
        
        prob = self.rng.uniform(0, 1)
        for key in self.testset_distribution:
            if self.testset_distribution[key] <= prob:
                return key
    
    def _filter_context(self, context: str):
        
        prompt = ChatPromptTemplate.from_messages(SCORE_CONTEXT.format(context=context))
        results = generate(prompts=[prompt], llm=self.filter_llm)
        score = eval(results.generations[0].text.strip())
        assert isinstance(score, float), "Score should be of type float"
        return score >= self.threshold
    
    
    def _seed_question(self, context: str):
        
        prompt = ChatPromptTemplate.from_messages(SEED_QUESTION.format(context=context))
        results = generate(prompts=[prompt], llm=self.generator_llm)
        return results.generations[0].text.strip()
    
    def _filter_question(self, question: str):
        
        prompt = ChatPromptTemplate.from_messages(FILTER_QUESTION.format(question=question))
        results = generate(prompts=[prompt], llm=self.filter_llm)
        return bool(results.generations[0].strip().endswith("Yes."))
    
    def _reasoning_question(self, question: str, context:str):
        
        return self._question_deepening(
            REASONING_QUESTION, question, context
        )
    
    def _condition_question(self, question: str, context: str):
        
        return self._question_deepening(
            CONDITIONAL_QUESTION, question, context
        )
        
    def _multicontext_question(self, question: str, context1: str, context2: str):
        
        prompt =  ChatPromptTemplate.from_messages(MULTICONTEXT_QUESTION.format(question=question, 
                                              context1=context1,
                                              context2=context2))
        results = generate(prompts=[prompt], llm=self.filter_llm)
        return results.generations[0].text.strip()
    
    def _compress_question(self, question: str):
        return self._question_transformation(COMPRESS_QUESTION, question=question)
        
    def _conversational_question(self, question: str):
        return self._question_transformation(CONVERSATION_QUESTION, question=question)

    def _question_transformation(self, prompt, question: str):
        
        prompt =  ChatPromptTemplate.from_messages(prompt.format(question=question))
        results = generate(prompts=[prompt], llm=self.filter_llm)
        return results.generations[0].text.strip()
    
    def _question_deepening(self, prompt, question, context):
        prompt =  ChatPromptTemplate.from_messages(prompt.format(question=question, context=context))
        results = generate(prompts=[prompt], llm=self.filter_llm)
        return results.generations[0].text.strip()
    
    def _remove_index(self, available_indices: list, node_idx: list):
        
        _ = [available_indices.pop(idx) for idx in node_idx]
        return available_indices
    
    def _generate_doc_node_map(self, documenet_nodes: t.List[TextNode]):
        
        doc_nodeidx = defaultdict(list)
        for idx, node in enumerate(documenet_nodes):
            doc_nodeidx[node.id_].update(idx)
            
        return doc_nodeidx
        
    def _get_neighbour_node(self, idx: int, node_indices: list):
        
        return [idx-1, idx] if idx == node_indices[-1] else [idx, idx+1]
        
        
    def _embed_nodes(self, nodes: t.List[TextNode]):
        
        embeddings = {}
        for node in nodes:
            embeddings[node.id_].update(self.embedding_model.embed_query(node.get_context()))
            
        return embeddings
        
    def generate(self, documents: t.List[Document], test_size: int):
        
        node_parser = SimpleNodeParser.from_defaults(chunk_size=self.chunk_size,
                                                     chunk_overlap=0,
                                                     include_metadata=True)
        document_nodes = node_parser.get_nodes_from_documents(documents=documents)

        if test_size > len(document_nodes):
            raise ValueError("""Maximum possible number of samples exceeded, 
                             reduce test_size or add more documents""")
        
        available_indices = np.arange(0, len(document_nodes)).tolist()
        doc_nodeidx = self._generate_doc_node_map(document_nodes)
        count = 0
        #TODO : Add progess bar
        while count < test_size and available_indices != []:
            
            size = self.rng.integers(1, 3)
            node_idx = self.rng.choice(available_indices, size=1)[0]
            available_indices = self._remove_index(available_indices, node_idx)

            neighbor_nodes = doc_nodeidx[document_nodes[node_idx].id_]
            node_indices = self._get_neighbour_node(node_idx, neighbor_nodes) if size > 1 else [node_idx]
            
            nodes = [document_nodes[node_idx] for node_idx in node_indices]
            text_chunk = " ".join([node.get_content() for node in nodes])
            score = self._filter_context(text_chunk)
            if not score:
                continue
            seed_question = self._seed_question(text_chunk)
            evolve_type = self._get_evolve_type()
            
            if evolve_type == "multicontext":
             
                node_embedding = self._embed_nodes([nodes[-1]])
                neighbor_nodes = self._remove_index(neighbor_nodes, node_indices)
                neighbour_emb = self._embed_nodes([document_nodes[idx] for idx in neighbor_nodes])
                _, indices = get_top_k_embeddings(node_embedding, neighbour_emb, similarity_cutoff=self.threshold)
                if indices:
                    best_neighbour = neighbor_nodes[indices[0]]
                question = self._multicontext_question(question=seed_question, 
                                                       context1=text_chunk,
                                                       context2=best_neighbour.get_content())
                text_chunk = "\n".join([text_chunk, best_neighbour.get_context()])

            else:    
                
                evolve_fun = question_deep_map.get(evolve_type)
                question = getattr(self, evolve_fun)(seed_question, text_chunk) if evolve_fun else seed_question
            
            
            
            
            
            
            
        