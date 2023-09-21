import typing as t
from collections import defaultdict, namedtuple

import numpy as np
import numpy.testing as npt
import pandas as pd
from attr import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain.prompts import ChatPromptTemplate
from llama_index import OpenAIEmbedding
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.readers.schema.base import Document
from llama_index.schema import TextNode
from numpy.random import default_rng

from ragas.metrics.llms import generate
from ragas.testset.prompts import (
    ANSWER_FORMULATE,
    COMPRESS_QUESTION,
    CONDITIONAL_QUESTION,
    CONTEXT_FORMULATE,
    CONVERSATION_QUESTION,
    FILTER_QUESTION,
    MULTICONTEXT_QUESTION,
    REASONING_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
)

DEFAULT_TESTDISTRIBUTION = {
    "simple": 0.5,
    "reasoning": 0.2,
    "multi_context": 0.2,
    "conditional": 0.1,
}

question_deep_map = {
    "reasoning": "_reasoning_question",
    "conditional": "_condition_question",
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

    def __init__(
        self,
        generator_llm: BaseLLM | BaseChatModel,
        ctitic_llm: BaseLLM | BaseChatModel,
        embeddings_model: Embeddings,
        testset_distribution: t.Optional[t.Dict[str, float]] = None,
        chat_qa: bool = True,
        chunk_size: int = 1024,
        seed: int = 42,
    ) -> None:
        self.generator_llm = generator_llm
        self.ctitic_llm = ctitic_llm
        self.embedding_model = embeddings_model
        testset_distribution = testset_distribution or DEFAULT_TESTDISTRIBUTION
        npt.assert_almost_equal(
            1,
            sum(testset_distribution.values()),
            err_msg="Sum of distribution should be 1",
        )

        probs = np.cumsum(list(testset_distribution.values()))
        types = testset_distribution.keys()
        self.testset_distribution = dict(zip(types, probs))

        self.chat_qa = chat_qa
        self.chunk_size = chunk_size
        self.threshold = 7.5
        self.rng = default_rng(seed)

    @classmethod
    def from_default(
        cls,
        openai_generator_llm: str = "gpt-3.5-turbo-16k",
        openai_filter_llm: str = "gpt-4",
    ):
        generator_llm = ChatOpenAI(model_name=openai_generator_llm)
        ctitic_llm = ChatOpenAI(model_name=openai_filter_llm)
        embeddings_model = OpenAIEmbedding()
        return cls(
            generator_llm=generator_llm,
            ctitic_llm=ctitic_llm,
            embeddings_model=embeddings_model,
        )

    def _get_evolve_type(self):
        prob = self.rng.uniform(0, 1)
        for key in self.testset_distribution.keys():
            if prob <= self.testset_distribution[key]:
                return key

    def _filter_context(self, context: str):
        human_prompt = SCORE_CONTEXT.format(context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = generate(prompts=[prompt], llm=self.ctitic_llm)
        score = eval(results.generations[0][0].text.strip())
        assert isinstance(score, float), "Score should be of type float"
        return score >= self.threshold

    def _seed_question(self, context: str):
        human_prompt = SEED_QUESTION.format(context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = generate(prompts=[prompt], llm=self.generator_llm)
        return results.generations[0][0].text.strip()

    def _filter_question(self, question: str):
        prompt = ChatPromptTemplate.from_messages(
            FILTER_QUESTION.format(question=question)
        )
        results = generate(prompts=[prompt], llm=self.ctitic_llm)
        return bool(results.generations[0][0].strip().endswith("Yes."))

    def _reasoning_question(self, question: str, context: str):
        return self._qc_template(REASONING_QUESTION, question, context)

    def _condition_question(self, question: str, context: str):
        return self._qc_template(CONDITIONAL_QUESTION, question, context)

    def _multicontext_question(self, question: str, context1: str, context2: str):
        human_prompt = MULTICONTEXT_QUESTION.format(
            question=question, context1=context1, context2=context2
        )
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = generate(prompts=[prompt], llm=self.generator_llm)
        return results.generations[0][0].text.strip()

    def _compress_question(self, question: str):
        return self._question_transformation(COMPRESS_QUESTION, question=question)

    def _conversational_question(self, question: str):
        return self._question_transformation(CONVERSATION_QUESTION, question=question)

    def _question_transformation(self, prompt, question: str):
        human_prompt = prompt.format(question=question)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = generate(prompts=[prompt], llm=self.generator_llm)
        return results.generations[0][0].text.strip()

    def _qc_template(self, prompt, question, context):
        human_prompt = prompt.format(question=question, context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = generate(prompts=[prompt], llm=self.generator_llm)
        return results.generations[0][0].text.strip()

    def _generate_answer(self, question: str, context: list[str]):
        return [
            self._qc_template(ANSWER_FORMULATE, qstn, context[i])
            for i, qstn in enumerate(question.split("\n"))
        ]

    def _generate_context(self, question: str, text_chunk: str):
        return [
            self._qc_template(CONTEXT_FORMULATE, qstn, text_chunk)
            for qstn in question.split("\n")
        ]

    def _remove_index(self, available_indices: list, node_idx: list):
        for idx in node_idx:
            available_indices.remove(idx)
        return available_indices

    def _generate_doc_node_map(self, documenet_nodes: t.List[TextNode]):
        doc_nodeidx = defaultdict(list)
        for idx, node in enumerate(documenet_nodes):
            doc_nodeidx[node.id_].append(idx)

        return doc_nodeidx

    def _get_neighbour_node(self, idx: int, node_indices: list):
        return [idx - 1, idx] if idx == node_indices[-1] else [idx, idx + 1]

    def _embed_nodes(self, nodes: t.List[TextNode]):
        embeddings = {}
        for node in nodes:
            embeddings[node.id_].update(
                self.embedding_model.embed_query(node.get_context())
            )

        return embeddings

    def generate(self, documents: t.List[Document], test_size: int):
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=0, include_metadata=True
        )
        document_nodes = node_parser.get_nodes_from_documents(documents=documents)

        if test_size > len(document_nodes):
            raise ValueError(
                """Maximum possible number of samples exceeded, 
                             reduce test_size or add more documents"""
            )

        available_indices = np.arange(0, len(document_nodes)).tolist()
        doc_nodeidx = self._generate_doc_node_map(document_nodes)
        count = 0
        Testdata_tuple = namedtuple(
            "Testdata_tuple", ["question", "context", "answer", "question_type"]
        )
        samples = []

        # TODO : Add progess bar
        while count < test_size and available_indices != []:
            size = self.rng.integers(1, 3)
            node_idx = self.rng.choice(available_indices, size=1)[0]
            available_indices = self._remove_index(available_indices, [node_idx])

            neighbor_nodes = doc_nodeidx[document_nodes[node_idx].id_]
            node_indices = (
                self._get_neighbour_node(node_idx, neighbor_nodes)
                if size > 1
                else [node_idx]
            )

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
                neighbor_emb = self._embed_nodes(
                    [document_nodes[idx] for idx in neighbor_nodes]
                )
                _, indices = get_top_k_embeddings(
                    node_embedding, neighbor_emb, similarity_cutoff=self.threshold
                )
                if indices:
                    best_neighbor = neighbor_nodes[indices[0]]
                question = self._multicontext_question(
                    question=seed_question,
                    context1=text_chunk,
                    context2=best_neighbor.get_content(),
                )
                text_chunk = "\n".join([text_chunk, best_neighbor.get_context()])

            else:
                evolve_fun = question_deep_map.get(evolve_type)
                question = (
                    getattr(self, evolve_fun)(seed_question, text_chunk)
                    if evolve_fun
                    else seed_question
                )

            if evolve_type != "simple":
                prob = self.rng.uniform(0, 1)
                if self.chat_qa and prob >= 0.5:
                    question = self._conversational_question(question=question)
                else:
                    question = self._compress_question(question=question)

            context = self._generate_context(question, text_chunk)
            answer = self._generate_answer(question, context)
            samples.append(
                Testdata_tuple(question.split("\n"), context, answer, evolve_type)
            )
            count += 1

        return TestDataset(test_data=samples)


@dataclass
class TestDataset:
    """
    TestDataset class
    """

    test_data: t.Sequence[tuple[list[str], list[str], list[str], str]]

    def to_pandas(self):
        data_samples = []
        for data in self.test_data:
            is_conv = len(data.context) > 1
            question_type = data.question_type
            data = [
                {
                    "question": qstn,
                    "context": ctx,
                    "answer": ans,
                    "question_type": question_type,
                    "episode_done": True,
                }
                for qstn, ctx, ans in zip(data.question, data.context, data.answer)
            ]
            if is_conv:
                data[-1] = data[0].update({"episode_done": False})
            data_samples.extend(data)

        return pd.DataFrame.from_records(data_samples)
