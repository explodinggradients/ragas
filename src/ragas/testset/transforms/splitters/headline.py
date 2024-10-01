import typing as t
from dataclasses import dataclass

from ragas.testset.graph import Node, NodeType, Relationship
from ragas.testset.transforms.base import Splitter


@dataclass
class HeadlineSplitter(Splitter):
    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        text = node.get_property("page_content")
        if text is None:
            raise ValueError("'page_content' property not found in this node")

        headlines = node.get_property("headlines")
        if headlines is None:
            raise ValueError("'headlines' property not found in this node")

        # create the chunks for the different sections
        indices = []
        for headline in headlines:
            indices.append(text.find(headline))
        indices.append(len(text))
        chunks = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]

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
