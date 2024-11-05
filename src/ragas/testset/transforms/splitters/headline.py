import typing as t
from dataclasses import dataclass

from ragas.testset.graph import Node, NodeType, Relationship
from ragas.testset.transforms.base import Splitter
from ragas.utils import num_tokens_from_string


@dataclass
class HeadlineSplitter(Splitter):
    min_tokens: int = 300

    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        text = node.get_property("page_content")
        if text is None:
            raise ValueError("'page_content' property not found in this node")

        headlines = node.get_property("headlines")
        if headlines is None:
            raise ValueError("'headlines' property not found in this node")

        # create the chunks for the different sections
        indices = [0]
        for headline in headlines:
            index = text.find(headline)
            if index != -1:
                indices.append(index)
        indices.append(len(text))
        chunks = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]
        # merge chunks if their length is less than 300 tokens
        merged_chunks = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if num_tokens_from_string(current_chunk) < self.min_tokens:
                current_chunk = "\n\n".join([current_chunk, next_chunk])
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk

        merged_chunks.append(current_chunk)
        chunks = merged_chunks

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
