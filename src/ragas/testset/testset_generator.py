from __future__ import annotations

import typing as t
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass

import numpy as np
import numpy.testing as npt
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document as LangchainDocument
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.readers.schema import Document as LlamaindexDocument
from llama_index.schema import BaseNode
from numpy.random import default_rng
from tqdm import tqdm

from ragas.llms import LangchainLLM
from ragas.testset.prompts import (
    ANSWER_FORMULATE,
    COMPRESS_QUESTION,
    CONDITIONAL_QUESTION,
    CONTEXT_FORMULATE,
    CONVERSATION_QUESTION,
    EVOLUTION_ELIMINATION,
    FILTER_QUESTION,
    INFORMAL_QUESTION,
    MULTICONTEXT_QUESTION,
    REASONING_QUESTION,
    REWRITE_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
    TABLE_QA,
)
from ragas.testset.utils import load_as_json

DEFAULT_TEST_DISTRIBUTION = {
    "simple": 0.4,
    "reasoning": 0.3,
    "multi_context": 0.0,
    "conditional": 0.3,
}

question_deep_map = {
    "reasoning": "_reasoning_question",
    "conditional": "_condition_question",
}

DataRow = namedtuple(
    "DataRow", ["seed_question", "question", "context", "answer", "question_type", "evolution_elimination"]
)


@dataclass
class TestDataset:
    """
    TestDataset class
    """

    test_data: t.List[DataRow]

    def to_pandas(self) -> pd.DataFrame:
        data_samples = []
        for data in self.test_data:
            is_conv = len(data.context) > 1
            question_type = data.question_type
            data = [
                {
                    "seed_question": seed,
                    "question": qstn,
                    "context": ctx,
                    "answer": ans,
                    "question_type": question_type,
                    "episode_done": True,
                    "evolution_elimination": data.evolution_elimination,
                }
                for seed, qstn, ctx, ans in zip(
                    data.seed_question, data.question, data.context, data.answer
                )
            ]
            if is_conv:
                data[0].update({"episode_done": False})
            data_samples.extend(data)

        return pd.DataFrame.from_records(data_samples)


class TestsetGenerator:

    """
    Ragas Test Set Generator

    Attributes
    ----------
    generator_llm: LangchainLLM
        LLM used for all the generator operations in the TestGeneration paradigm.
    critique_llm: LangchainLLM
        LLM used for all the filtering and scoring operations in TestGeneration
        paradigm.
    embeddings_model: Embeddings
        Embeddings used for vectorizing nodes when required.
    chat_qa: float
        Determines the fraction of conversational questions the resulting test set.
    chunk_size: int
        The chunk size of nodes created from data.
    test_distribution : dict
        Distribution of different types of questions to be generated from given
        set of documents. Defaults to {"easy":0.1, "reasoning":0.4, "conversation":0.5}
    """

    def __init__(
        self,
        generator_llm: LangchainLLM,
        critic_llm: LangchainLLM,
        embeddings_model: Embeddings,
        testset_distribution: t.Optional[t.Dict[str, float]] = None,
        chat_qa: float = 0.0,
        chunk_size: int = 356,
        seed: int = 42,
    ) -> None:
        self.generator_llm = generator_llm
        self.critic_llm = critic_llm
        self.embedding_model = embeddings_model
        testset_distribution = testset_distribution or DEFAULT_TEST_DISTRIBUTION
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
        self.max_fixes = 2
        self.rng = default_rng(seed)

    @classmethod
    def from_default(
        cls,
        openai_generator_llm: str = "gpt-3.5-turbo-16k",
        openai_filter_llm: str = "gpt-4",
        chat_qa: float = 0.3,
        chunk_size: int = 512,
        testset_distribution: dict = DEFAULT_TEST_DISTRIBUTION,
    ):
        generator_llm = LangchainLLM(llm=ChatOpenAI(model=openai_generator_llm))
        critic_llm = LangchainLLM(llm=ChatOpenAI(model=openai_filter_llm))
        embeddings_model = OpenAIEmbeddings()  # type: ignore
        return cls(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings_model=embeddings_model,
            chat_qa=chat_qa,
            chunk_size=chunk_size,
            testset_distribution=testset_distribution,
        )

    def _get_evolve_type(self) -> str:
        """
        Decides question evolution type based on probability
        """
        prob = self.rng.uniform(0, 1)
        return next(
            (
                key
                for key in self.testset_distribution.keys()
                if prob <= self.testset_distribution[key]
            ),
            "simple",
        )

    def _filter_context(self, context: str) -> t.Dict:
        """
        context: str
            The input context

        Checks if the context is has information worthy of framing a question
        """
        human_prompt = SCORE_CONTEXT.format(context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.critic_llm.generate(prompts=[prompt])
        output = results.generations[0][0].text.strip()
        output = load_as_json(output)
        output.update({"score": output.get("score", 0) >= self.threshold})
        return output

    def _seed_question(self, context: str, is_table_present: bool) -> str:
        if is_table_present:
            human_prompt = TABLE_QA.format(context=context)
        else:
            human_prompt = SEED_QUESTION.format(context=context)

        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _filter_question(self, question: str) -> bool:
        human_prompt = FILTER_QUESTION.format(question=question)
        prompt = ChatPromptTemplate.from_messages([human_prompt])

        results = self.critic_llm.generate(prompts=[prompt])
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        print(json_results)
        return json_results.get("verdict") != "No"
    
    def _rewrite_question(self, question: str, context: str) -> str:
        
        human_prompt = REWRITE_QUESTION.format(question=question, context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])

        results = self.generator_llm.generate(prompts=[prompt])
        results = results.generations[0][0].text.strip()
        return results
        
    
    def _evolution_elimination(self, question1: str, question2: str) -> bool:
        
        human_prompt = EVOLUTION_ELIMINATION.format(question1=question1, question2=question2)
        prompt = ChatPromptTemplate.from_messages([human_prompt])

        results = self.critic_llm.generate(prompts=[prompt])
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        return json_results.get("verdict") != "Not Equal"

    def _reasoning_question(self, question: str, context: str) -> str:
        return self._qc_template(REASONING_QUESTION, question, context)

    def _condition_question(self, question: str, context: str) -> str:
        return self._qc_template(CONDITIONAL_QUESTION, question, context)

    def _multicontext_question(
        self, question: str, context1: str, context2: str
    ) -> str:
        human_prompt = MULTICONTEXT_QUESTION.format(
            question=question, context1=context1, context2=context2
        )
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _compress_question(self, question: str) -> str:
        return self._question_transformation(COMPRESS_QUESTION, question=question)

    def _conversational_question(self, question: str) -> str:
        return self._question_transformation(CONVERSATION_QUESTION, question=question)

    def _question_transformation(self, prompt, question: str) -> str:
        human_prompt = prompt.format(question=question)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _qc_template(self, prompt, question, context) -> str:
        human_prompt = prompt.format(question=question, context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _generate_answer(self, question: str, context: list[str]) -> t.List[str]:
        return [
            self._qc_template(ANSWER_FORMULATE, qstn, context[i])
            for i, qstn in enumerate(question.split("\n"))
        ]

    def _generate_context(self, question: str, text_chunk: str) -> t.List[str]:
        return [
            self._qc_template(CONTEXT_FORMULATE, qstn, text_chunk)
            for qstn in question.split("\n")
        ]
        
    def _question_transform(self, question:str) -> str:
        
        output = []
        for qstn in question.split('\n'):
            human_prompt = INFORMAL_QUESTION.format(question=qstn)
            prompt = ChatPromptTemplate.from_messages([human_prompt])
            results = self.generator_llm.generate(prompts=[prompt])
            output.append(results.generations[0][0].text.strip())
        
        return '\n'.join(output)
        

    def _remove_nodes(
        self, available_indices: list[BaseNode], node_idx: list
    ) -> t.List[BaseNode]:
        for idx in node_idx:
            available_indices.remove(idx)
        return available_indices

    def _generate_doc_nodes_map(
        self, document_nodes: t.List[BaseNode]
    ) -> t.Dict[str, t.List[BaseNode]]:
        doc_nodes_map: t.Dict[str, t.List[BaseNode]] = defaultdict(list[BaseNode])
        for node in document_nodes:
            file_name = node.metadata.get('file_name')
            if file_name:
                doc_nodes_map[file_name].append(node)

        return doc_nodes_map  # type: ignore

    def _get_neighbour_node(
        self, node: BaseNode, related_nodes: list[BaseNode], max_tokens=1000, after: bool=True,
    ) -> t.List[BaseNode]:
        if len(related_nodes) < 2:
            warnings.warn("No neighbors exists")
            return [node]
        idx = related_nodes.index(node)
        tokens = 0
        nodes = []
        inc = 1 if after else -1
        while tokens < max_tokens and idx>=0 and idx<len(related_nodes):
            nodes.append(related_nodes[idx])
            idx += inc
            #TODO: replace split with tikitoken
            tokens += len(related_nodes[idx].get_content().split())
                
        return nodes if after else nodes[::-1]

    def _embed_nodes(self, nodes: t.List[BaseNode]) -> t.Dict[str, t.List[float]]:
        embeddings = {}
        for node in nodes:
            embeddings[node.id_] = list(
                self.embedding_model.embed_query(node.get_content())
            )

        return embeddings

    def generate(
        self,
        documents: list[LlamaindexDocument] | list[LangchainDocument],
        test_size: int,
    ) -> TestDataset:
        if not isinstance(documents[0], (LlamaindexDocument, LangchainDocument)):
            raise ValueError(
                "Testset Generatation only supports LlamaindexDocuments or LangchainDocuments"  # noqa
            )

        if isinstance(documents[0], LangchainDocument):
            # cast to LangchainDocument since its the only case here
            documents = t.cast(t.List[LangchainDocument], documents)
            documents = [
                LlamaindexDocument.from_langchain_format(doc) for doc in documents
            ]
        # Convert documents into nodes
        # TODO: modify this to 
        # each node should contain docs of preffered chunk size
        # append document to provide enough context 
        # Use metadata for this.
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=0, include_metadata=True
        )
        documents = t.cast(t.List[LlamaindexDocument], documents)
        document_nodes: t.List[BaseNode] = node_parser.get_nodes_from_documents(
            documents=documents
        )
        # maximum 1 seed question per node
        if test_size > len(document_nodes):
            raise ValueError(
                """Maximum possible number of samples exceeded, 
                             reduce test_size or add more documents"""
            )

        available_nodes = document_nodes
        doc_nodes_map = self._generate_doc_nodes_map(document_nodes)
        count = 0
        samples = []

        pbar = tqdm(total=test_size)
        while count < test_size and available_nodes != []:
            evolve_type = self._get_evolve_type()
            curr_node = self.rng.choice(np.array(available_nodes), size=1)[0]
            available_nodes = self._remove_nodes(available_nodes, [curr_node])

            neighbor_nodes = doc_nodes_map[curr_node.metadata['file_name']]

            # Append multiple nodes randomly to remove chunking bias
            if len(curr_node.get_content().split()) < self.chunk_size:
                size = self.chunk_size - len(curr_node.get_content().split())
                nodes = self._get_neighbour_node(curr_node, neighbor_nodes, max_tokens=size, after=False)
            else:
                nodes = [curr_node]
                
            text_chunk = "\n".join([node.get_content() for node in nodes])
            print("Len of text chunks", len(nodes), len(text_chunk.split()))
            context_filter = self._filter_context(text_chunk)
            if not context_filter.get("score"):
                continue
          
            is_table_qa = context_filter.get("is_table_present", False)
            seed_question = self._seed_question(text_chunk, is_table_qa)
            evolve_type = (
                "simple"
                if ((evolve_type == "multi_context") and (is_table_qa))
                else evolve_type
            )
            print("seed question", seed_question)
            is_valid_question = self._filter_question(seed_question)
            tries = 1
            
            while tries < self.max_fixes and not is_valid_question:
                nodes = self._get_neighbour_node(nodes[0], neighbor_nodes, max_tokens=500, after=False)
                text_chunk = "\n".join([node.get_content() for node in nodes])
                seed_question = self._rewrite_question(question=seed_question, context=text_chunk)
                print("rewritten question", seed_question)
                is_valid_question = self._filter_question(seed_question)
                tries += 1

            if not is_valid_question:
                continue
            
            if evolve_type == "multi_context":
                # Find most similar chunk in same document
                #TODO: handle cases where neighbour nodes is null, ie multi context across documents
                # First preference - nodes from same document, second preference - other docs
                node_embedding = self._embed_nodes([nodes[-1]])
                neighbor_nodes = self._remove_nodes(neighbor_nodes, nodes)
                neighbor_emb = self._embed_nodes(neighbor_nodes)

                _, indices = get_top_k_embeddings(
                    list(node_embedding.values())[0],
                    list(neighbor_emb.values()),
                    similarity_cutoff=self.threshold / 10,
                )
                if indices:
                    # type cast indices from list[Any] to list[int]
                    indices = t.cast(t.List[int], indices)
                    best_neighbor = neighbor_nodes[indices[0]]
                    question = self._multicontext_question(
                        question=seed_question,
                        context1=text_chunk,
                        context2=best_neighbor.get_content(),
                    )
                    text_chunk = "\n".join([text_chunk, best_neighbor.get_content()])
                else:
                    continue

            # for reasoning and conditional modes, evolve question with the
            # functions from question_deep_map
            else:
                evolve_fun = question_deep_map.get(evolve_type)
                question = (
                    getattr(self, evolve_fun)(seed_question, text_chunk)
                    if evolve_fun
                    else seed_question
                )

            # compress question or convert into conversational questions
            if evolve_type != "simple":
                prob = self.rng.uniform(0, 1)
                if self.chat_qa and prob <= self.chat_qa:
                    question = self._conversational_question(question=question)
                else:
                    question = self._compress_question(question=question)

            is_valid_question = self._filter_question(question) if evolve_type!="simple" else True
            if evolve_type != "simple":
                evolution_elimination = self._evolution_elimination(question1=seed_question,
                                                                 question2=question)
            else:
                evolution_elimination = None
                
            if is_valid_question:
                question = self._question_transform(question)
                context = self._generate_context(question, text_chunk)
                answer = self._generate_answer(question, context)
                samples.append(
                    DataRow(
                        [seed_question],
                        question.split("\n"),
                        context,
                        answer,
                        evolve_type,
                        evolution_elimination,
                    )
                )
                count += 1
                pbar.update(count)

        return TestDataset(test_data=samples)


if __name__ == "__main__":
    from llama_index import SimpleDirectoryReader

    reader = SimpleDirectoryReader("/Users/shahules/belar/experimental/arxiv-papers/", num_files_limit=10)
    documents = reader.load_data()
    testsetgenerator = TestsetGenerator.from_default(chunk_size=312)
    test_size = 10
    testset = testsetgenerator.generate(documents, test_size=test_size)
