import typing as t
from dataclasses import dataclass
import tiktoken

from ragas.testset.graph import Node, NodeType, Relationship
from ragas.testset.transforms.base import Splitter
from ragas.utils import num_tokens_from_string


DEFAULT_TOKENIZER = tiktoken.get_encoding("o200k_base")


@dataclass
class HeadlineSplitter(Splitter):
    min_tokens: int = 300
    max_tokens: int = 1000

    def adjust_chunks(self, chunks):
        adjusted_chunks = []
        current_chunk = ""

        for chunk in chunks:
            chunk_tokens = DEFAULT_TOKENIZER.encode(chunk)

            # Split chunks that are over max_tokens
            while len(chunk_tokens) > self.max_tokens:
                adjusted_chunks.append(DEFAULT_TOKENIZER.decode(chunk_tokens[:self.max_tokens]))
                chunk_tokens = chunk_tokens[self.max_tokens:]

            chunk_str = DEFAULT_TOKENIZER.decode(chunk_tokens)
            if len(chunk_tokens) < self.min_tokens:
                if current_chunk:
                    current_chunk += " " + chunk_str
                    if num_tokens_from_string(current_chunk, encoding_name=DEFAULT_TOKENIZER.name) >= self.min_tokens:
                        adjusted_chunks.append(current_chunk)
                        current_chunk = ""
                else:
                    current_chunk = chunk_str
            else:
                if current_chunk:
                    adjusted_chunks.append(current_chunk)
                    current_chunk = ""
                adjusted_chunks.append(chunk_str)

        # Append any remaining chunk
        if current_chunk:
            adjusted_chunks.append(current_chunk)

        return adjusted_chunks

    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        text = node.get_property("page_content")
        if text is None:
            raise ValueError("'page_content' property not found in this node")

        headlines = node.get_property("headlines")
        if headlines is None:
            raise ValueError("'headlines' property not found in this node")

        if num_tokens_from_string(text, encoding_name=DEFAULT_TOKENIZER.name) < self.min_tokens:
            return [node], []
        # create the chunks for the different sections
        indices = [0]
        for headline in headlines:
            index = text.find(headline)
            if index != -1:
                indices.append(index)
        indices.append(len(text))
        chunks = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]
        chunks = self.adjust_chunks(chunks)

        # if there was no headline, return the original node
        if len(chunks) == 1:
            return [node], []

        # create the nodes
        nodes = [
            Node(type=NodeType.CHUNK, properties={"page_content": chunk})
            for chunk in chunks
        ]

        # create the relationships for children
        relationships = []
        for child_node in nodes:
            relationships.append(
                Relationship(
                    type="child",
                    source=node,
                    target=child_node,
                )
            )

        # create the relationships for the next nodes
        for i, child_node in enumerate(nodes):
            if i < len(nodes) - 1:
                relationships.append(
                    Relationship(
                        type="next",
                        source=child_node,
                        target=nodes[i + 1],
                    )
                )
        return nodes, relationships
