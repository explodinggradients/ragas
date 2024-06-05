from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import metadata
import typing as t
from graphene.types.schema import Schema
import numpy as np
from langchain_core.documents import Document as LCDocument

from ragas.llms.base import BaseRagasLLM
from ragas.testsetv3.graph import Node, Relationship
from ragas.testsetv3.utils import rng

from ragas.testsetv3.query_prompts import common_themes_from_summaries, abstract_question_from_theme, critic_question 

from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.retrievers.document_compressors import FlashrankRerank

@dataclass
class QueryGenerator(ABC):
    llm: BaseRagasLLM
    schema: Schema
    nodes: t.List[Node]
    relationships: t.List[Relationship]
    
    
    @abstractmethod
    def generate_question(self, query: str) -> t.Any:
        pass
        
    @abstractmethod
    def critic_question(self, query: str) -> bool:
        pass
    
    @abstractmethod
    def generate_answer(self, query: str) -> t.Any:
        pass
    
@dataclass    
class AbstractQueries(QueryGenerator):
    
    document_compressor: BaseDocumentCompressor = FlashrankRerank()
    
    async def generate_question(self, query: str) -> t.Any:
        
        
        # query for document nodes that are related to other document nodes with similarity relationship
        query = '''
        {
        filterNodes(label: DOC) {
            id
            label
            properties
            relationships(label: "similar_summary") {
            label
            properties
            target {
                id
                label
                properties
            }
            }
        }
        }
        '''
        results = self.schema.execute(query, context={"nodes": self.nodes, "relationships": self.relationships})
        if not results.data:
            return None
        
        result_nodes = [Node(**item) for item in result.data['filterNodes']]
        # replace by query
        # nodes_ = [Node(**relation["target"]) for node in result_nodes for relation in node.relationships]
        
        nodes_weights = np.array([node.properties["chances"] for node in nodes_])
        nodes_weights = nodes_weights / sum(nodes_weights)
        current_nodes = rng.choice(np.array(nodes_), p=nodes_weights, size=1).tolist()
        
        # query for nodes that are related to current node using specific relationships
        # results = self.schema.execute(query, context={"nodes": [current_node], "relationships": current_node.relationships})
        # node_ids = [item.get("id") for item in results.data[""]]
        # nodes_ = []
        related_nodes = [rel.target for rel in current_nodes[0].relationships if rel.target.label == "DOC"]
        current_nodes.append(related_nodes)
        summaries = [current_node.properties["summary"] for current_node in current_nodes]
        common_themes = await self.llm.generate(common_themes_from_summaries.format(text=summaries))
        output = await self.llm.generate(abstract_question_from_theme.format(theme=common_themes))
        abstract_questions = output.generations[0][0].text
        critic_verdict = [await self.critic_question(question) for question in abstract_questions]
        abstract_questions = [question for question, verdict in zip(abstract_questions, critic_verdict) if verdict]
        return abstract_questions
        
    async def critic_question(self, query: str) -> bool:
        
        output = await self.llm.generate(critic_question.format(text=query))
        output = output.generations[0][0].text
        return all(score > 2 for score in output.values())
        
    async def generate_answer(self, question: str, nodes: t.List[Node]) -> t.Any:
        
        # query to get all child nodes of nodes
        nodes = [item.target for node in nodes for item in node.relationships if item.label == "contains"]
        chunks = [LCDocument(page_content=node.properties["page_content"], metadata=node.properties["metadata"]) for node in nodes]
        ranked_chunks = self.document_compressor.compress_documents(documents=chunks, query=question)
        ranked_chunks = [chunk.page_content for chunk in ranked_chunks]
        model_name = self.llm.langchain_llm.model_name or "gpt-2"
        enc = tiktoken.encoding_for_model(model_name)
        ranked_chunks_length = [len(enc.encode(chunk)) for chunk in ranked_chunks]
        ranked_chunks_length = np.cumsum(ranked_chunks_length)
        max_tokens = 2000
        index = np.argmax(ranked_chunks_length < max_tokens) + 1
        chunks = ranked_chunks[:index]
        return await self.llm.generate(question_answering.format(question=question, text=chunks))
        
        
        
        
        
        #embedding search in resulting nodes to identify chunks 
        
        
    