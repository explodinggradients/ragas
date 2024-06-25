import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
import tiktoken
from langchain.utils.math import cosine_similarity, cosine_similarity_top_k
from langchain_core.documents import Document as LCDocument

from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas.testsetv3.graph import Node, NodeLevel
from ragas.testsetv3.questions.base import QAC, QAGenerator, DEFAULT_DISTRIBUTION
from ragas.testsetv3.questions.prompts import (
    abstract_question_from_theme,
    common_theme_from_summaries,
    common_topic_from_keyphrases,
    comparative_question,
    critic_question,
    question_answering,
)
from ragas.testsetv3.questions.queries import CLUSTER_OF_RELATED_NODES_QUERY, LEAF_NODE_QUERY
from ragas.executor import Executor
from ragas.testsetv3.utils import rng


logger = logging.getLogger(__name__)


class AbstractQANew(QAGenerator):
    
    
    def query_nodes(self, query: str, kwargs) -> t.Any:
        return super().query_nodes(query, kwargs)
    
    def filter_nodes(self, query: str, kwargs) -> t.Any:
        results = self.query_nodes(query, kwargs)
        sets = []
        for cluster in results:
            ids = [cluster.id]
            ids.extend([rel.target.id for rel in cluster.relationships])
            sets.append(set(ids))
        
        sets = np.unique(sets)
        # Convert tuples to sets for easy comparison
        sets_as_sets = [set(s) for s in sets]
        supersets = []
        for s in sets_as_sets:
            is_subset = False
            for other in sets_as_sets:
                if other != s and other.issuperset(s):
                    is_subset = True
                    break
            if not is_subset:
                supersets.append(s)
            
        nodes_to_return = []    
        for subset in supersets:
            nodes_to_return.append([node for node in self.nodes if node.id in subset])
            
        return nodes_to_return or None
    
    async def critic_question(self, question: str):
        raise NotImplementedError("critic_question is not implemented")
    
    async def generate_answer(self, question: str, chunks: t.List[LCDocument]):
        raise NotImplementedError("generate_answer is not implemented")
    
    async def generate_question(self, query: str | None = None, kwargs: dict | None = None):
        raise NotImplementedError("generate_question is not implemented")


@dataclass
class AbtractQA(AbstractQANew):
    name: str = "AbstractQA"
    distribution = DEFAULT_DISTRIBUTION
    generate_question_prompt: Prompt = field(
        default_factory=lambda: abstract_question_from_theme
    )
    critic_question_prompt: Prompt = field(default_factory=lambda: critic_question)
    generate_answer_prompt: Prompt = field(default_factory=lambda: question_answering)
    generate_common_theme_prompt: Prompt = field(
        default_factory=lambda: common_theme_from_summaries
    )
    
    async def run(self, query, kwargs):
        
        query = query or CLUSTER_OF_RELATED_NODES_QUERY
        if kwargs is None:
            kwargs = {
                "label": "summary_similarity",
                "property": "score",
                "value": 0.5,
                "comparison": "gt",
            }
            
        node_clusters = self.filter_nodes(query, kwargs)
        if node_clusters is None:
            logging.warning("No nodes satisfy the query")
            return QAC()

        num_clusters = min(self.num_samples, len(node_clusters))
        seed_per_results = self.num_samples // len(node_clusters)
        reminder = self.num_samples - seed_per_results * num_clusters
        seeds = [seed_per_results] * num_clusters
        seeds[-1] += reminder
        
        nodes_themes = []
        for cluster, num_seeds in zip(node_clusters, seeds):
              
            summaries = [item.properties["metadata"]["summary"] for item in cluster]
            summaries = "\n".join(
                [f"summary {i+1}: {summary}" for i, summary in enumerate(summaries)]
            )
            common_themes = await self.llm.generate(
                self.generate_common_theme_prompt.format(summaries=summaries, num_themes=num_seeds)
            )
            common_themes = await json_loader.safe_load(common_themes.generations[0][0].text, llm=self.llm)
            nodes_themes.extend([(cluster, theme) for theme in common_themes])
            
            
        exec = Executor(
            desc="Generating",
            keep_progress_bar=True,
            raise_exceptions=True,
            run_config=None,
        )
        
        index = 0
        for distribution, prob in self.distribution.items():
            style, length = distribution
            for i in range(int(prob * self.num_samples)):
                exec.submit(self.generate_question,
                            nodes_themes[index][0], style, length, {"common_theme": nodes_themes[index][1]})
                index += 1
                

        remaining_size = self.num_samples - index    
        if remaining_size != 0:
            choices = np.array(self.distribution.keys())
            prob = np.array(self.distribution.values())
            random_distribution = rng.choice(choices, p=prob, size=remaining_size)
            for distribution in random_distribution:
                style, length = distribution
                exec.submit(self.generate_question,
                            nodes_themes[index][0], style, length, {"common_theme": nodes_themes[index][1]})
                index += 1
                
        return exec.results()

    async def generate_question(
        self, nodes, style, length, kwargs: t.Optional[dict] = None
    ) -> QAC:
        assert self.llm is not None, "LLM is not initialized"
        kwargs = kwargs or {}
        common_theme = kwargs.get("common_theme", "")
        try:
            source = await self.retrieve_chunks(nodes,{"max_tokens": 4024, "topic": common_theme.get("description")})
            if source:
                source_text = "\n\n".join([chunk.page_content for chunk in source])
                abstract_question = await self.llm.generate(
                    self.generate_question_prompt.format(
                        theme=common_theme["theme"], context=source_text
                    )
                )
                abstract_question = abstract_question.generations[0][0].text
                critic_verdict = await self.critic_question(abstract_question)
                if critic_verdict:
                    abstract_question = await self.modify_question(abstract_question, style, length)
                    answer = await self.generate_answer(abstract_question, source)
                    return QAC(
                    question=abstract_question,
                    answer=answer,
                    source=source,
                    name=self.name,
                    style=style,
                    length=length,
                )
                else:
                    logger.warning("Critic rejected the question: %s", abstract_question)
                    return QAC()
            else:
                logger.warning("source was not detected %s", common_theme)
                return QAC()
            
        except Exception as e:
            logger.error("Error while generating question: %s", e)
            raise e
        
    async def critic_question(self, question: str) -> bool:
        assert self.llm is not None, "LLM is not initialized"

        output = await self.llm.generate(critic_question.format(question=question))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    async def retrieve_chunks(
        self, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.List[LCDocument] | None:
        assert self.embedding is not None, "Embedding is not initialized"
        assert self.llm is not None, "LLLM is not initialized"

        kwargs = kwargs or {}
        max_tokens = kwargs.get("max_tokens", 3000)
        topic = kwargs.get("topic", "")
        node_ids = [node.id for node in nodes]
        
        query = LEAF_NODE_QUERY
        leaf_nodes = [self.query_nodes(query, {"id": json.dumps(id)}) for id in node_ids]
        leaf_nodes = [node for nodes in leaf_nodes for node in nodes]
        if leaf_nodes is None:
            return None
        
        output_documents = []
        #TODO: also filter summaries using cosine
        summaries = [f"{node.properties['metadata']['title']}: {node.properties['metadata']['summary']}" for node in nodes]
        output_documents.append(LCDocument(page_content="\n".join(summaries), metadata={"source": "summary"}))
        
        page_embeddings = [
            node.properties["metadata"]["page_content_embedding"] for node in leaf_nodes
        ]

        topic_embedding = await self.embedding.embed_text(topic)
        
        # TODO: replace with similarity top-k with threshold cutoff
        # print(cosine_similarity_top_k([topic_embedding], page_embeddings, score_threshold=0.7))
        similarity_matrix = cosine_similarity([topic_embedding], page_embeddings)
        most_similar = np.flip(np.argsort(similarity_matrix[0]))
        ranked_lead_nodes = [leaf_nodes[i] for i in most_similar]
        # TODO: allow for different models
        model_name = "gpt-3.5-turbo-"
        enc = tiktoken.encoding_for_model(model_name)
        ranked_chunks_length = [
            len(enc.encode(node.properties["page_content"])) for node in ranked_lead_nodes
        ]
        ranked_chunks_length = np.cumsum(ranked_chunks_length)
        index = np.argmax(np.argwhere(np.cumsum(ranked_chunks_length) < max_tokens)) + 1
        top_leaf_nodes = ranked_lead_nodes[:index]
        dict = {}
        
        for node in top_leaf_nodes:
            if node.properties["metadata"]["source"] in dict:
                dict[node.properties["metadata"]["source"]].append({node.level.value: node})
            else:
                dict[node.properties["metadata"]["source"]] = [{node.level.value: node}]          
        
        for source, nodes in dict.items():
            sorted_nodes = sorted(nodes, key=lambda x: list(x.keys())[0])
            dict[source] = sorted_nodes
            current_nodes = [elem for item in sorted_nodes for elem in item.values()]
            title, source_file = current_nodes[0].properties["metadata"].get("title"), current_nodes[0].properties["metadata"].get("source")
            text = "\n\n".join([node.properties["page_content"] for node in current_nodes])
            text = f"Document title: {title}\n\n{text}"
            output_documents.append(LCDocument(page_content=text, metadata={"source": source_file}))
        
        return output_documents

    async def generate_answer(
        self,
        question: str,
        chunks: t.List[LCDocument],
    ) -> str:
        assert self.llm is not None, "LLM is not initialized"
        # TODO : add title+summary of each node + title + content from most relevant chunk
        text = "\n\n".join([chunk.page_content for chunk in chunks])
        output = await self.llm.generate(
            self.generate_answer_prompt.format(question=question, text=text)
        )
        return output.generations[0][0].text


@dataclass
class ComparitiveAbtractQA(AbtractQA):
    name: str = "ComparitiveAbtractQA"
    common_topic_prompt: Prompt = field(
        default_factory=lambda: common_topic_from_keyphrases
    )
    generate_question_prompt: Prompt = field(
        default_factory=lambda: comparative_question
    )

    async def generate_question(
        self, query: t.Optional[str] = None, kwargs: t.Optional[dict] = None
    ) -> QAC:
        assert self.llm is not None, "LLM is not initialized"

        query = query or CLUSTER_OF_RELATED_NODES_QUERY
        
        kwargs = kwargs or {
            "label": "jaccard_over_keyphrases",
            "property": "score",
            "value": 0.2,
            "comparison": "gt",
        }
        
        result_nodes = self.query_nodes(query, kwargs)
        current_nodes = self.get_random_node(result_nodes)

        indices = np.flip(
            np.argsort(
                [rel.properties["score"] for rel in current_nodes[0].relationships]
            )
        )[:3]
        related_nodes = [current_nodes[0].relationships[i].target for i in indices]
        current_nodes.extend(related_nodes)

        summaries = [item.properties["metadata"]["summary"] for item in current_nodes]
        summaries = "\n".join(
            [f"{i+1}. {summary}" for i, summary in enumerate(summaries)]
        )
        common_theme = await self.llm.generate(
            self.generate_common_theme_prompt.format(summaries=summaries)
        )
        common_theme = common_theme.generations[0][0].text

        keyphrases = [
            node.properties["metadata"]["keyphrases"] for node in current_nodes
        ]
        keyphrases = [phrase for phrases in keyphrases for phrase in phrases]
        comparison_topic = await self.llm.generate(
            self.common_topic_prompt.format(theme=common_theme, keyphrases=keyphrases)
        )
        comparison_topic = json.loads(comparison_topic.generations[0][0].text)[0]

        question = await self.llm.generate(
            self.generate_question_prompt.format(
                theme=common_theme, topic=comparison_topic
            )
        )
        question = question.generations[0][0].text

        kwargs = {"max_tokens": 4024, "common_theme": common_theme}

        critic_verdict = await self.critic_question(question)
        if critic_verdict:
            source = await self.retrieve_chunks(question, current_nodes, kwargs)
            question = await self.modify_question(question)
            if source:
                answer = await self.generate_answer(question, source)
                return QAC(
                    question=question,
                    answer=answer,
                    source=source,
                    name=self.name,
                    style=self.style,
                    length=self.length,
                )
            else:
                logger.warning("source not found: %s", question)
                return QAC()
        else:
            logger.warning("Critic rejected the question: %s", question)
            return QAC()

    async def critic_question(self, question: str) -> bool:
        assert self.llm is not None, "LLM is not initialized"

        output = await self.llm.generate(critic_question.format(question=question))
        output = json.loads(output.generations[0][0].text)
        return all(score >= 2 for score in output.values())

    async def generate_answer(self, question: str, chunks: t.List[LCDocument]) -> str:
        assert self.llm is not None, "LLM is not initialized"

        text = "\n\n".join([chunk.page_content for chunk in chunks])
        output = await self.llm.generate(
            self.generate_answer_prompt.format(question=question, text=text)
        )
        return output.generations[0][0].text

    async def retrieve_chunks(
        self, question: str, nodes: t.List[Node], kwargs: t.Optional[dict] = None
    ) -> t.List[LCDocument] | None:
        kwargs = kwargs or {}
        assert self.embedding is not None, "Embedding is not initialized"

        common_theme = kwargs.get("common_theme", "")
        query_emebdding = await self.embedding.embed_text(common_theme)

        node_ids = [node.id for node in nodes]
        node_ids = json.dumps(node_ids)

        query = """
        {{
        filterNodes(ids: {node_ids}) {{
            id
            label
            properties
            relationships(label: "child") {{
            label
            properties
            target {{
                id
                label
                properties
                level
            }}
            }}
        }}
        }}
        """
        kwargs = {"node_ids": node_ids}
        target_nodes = self.query_nodes(query, kwargs)
        if not target_nodes:
            return None

        target_nodes = [
            relation.target for node in target_nodes for relation in node.relationships
        ]
        target_nodes = [
            node for node in target_nodes if node.level == NodeLevel.LEVEL_1.name
        ]
        context_embedding = [
            node.properties["metadata"]["page_content_embedding"]
            for node in target_nodes
        ]
        idxs, _ = cosine_similarity_top_k([query_emebdding], context_embedding, top_k=2)
        target_nodes = [target_nodes[idx[1]] for idx in idxs]
        documents = [
            LCDocument(
                page_content=node.properties["page_content"],
                metadata=node.properties["metadata"],
            )
            for node in target_nodes
        ]
        return documents
