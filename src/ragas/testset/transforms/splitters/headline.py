import re
import typing as t
from dataclasses import dataclass
from unidecode import unidecode

from ragas.testset.graph import Node, NodeType, Relationship
from ragas.testset.transforms.base import Splitter


def normalize_text(text):
    return unidecode(re.sub(r'\s+', '', text).lower())


def remove_indices(text):
    cleaned_text = re.sub(r'(\d+\.)+ *', '', text)
    return cleaned_text


def adjust_indices(original_text, indices):
    last_index = 0
    count = 0

    indices = sorted(indices)
    new_indices = []
    for index in indices:
        while last_index < len(original_text):
            if not original_text[last_index].isspace():
                count += 1
            if count == index + 1:
                new_indices.append(last_index)
                last_index += 1
                break
            last_index += 1
    
    return new_indices


@dataclass
class HeadlineSplitter(Splitter):
    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        text = node.get_property("page_content")
        if text is None:
            raise ValueError("'page_content' property not found in this node")

        headlines = node.get_property("headlines")
        if headlines is None:
            raise ValueError("'headlines' property not found in this node")

        if len(headlines) == 0:
            return [], []
        
        # create the chunks for the different sections
        indices = []
        normalized_text = normalize_text(text)

        for headline in headlines:
            if headline is not None and not headline.isspace():
                indice = normalized_text.find(normalize_text(headline))
                if indice == -1:
                    text_search = remove_indices(headline)
                    text_search = normalize_text(text_search)
                    indice = normalized_text.find(text_search)

                if indice != -1:
                    indices.append(indice)
        
        if len(indices) == 0:
            return [], []
        
        indices = adjust_indices(text, indices)

        indices.append(len(text))

        chunks = []
        for i in range(len(indices) - 1):
            aux = text[indices[i] : indices[i + 1]]

            if not aux.isspace():
                chunks.append(aux)

        if len(chunks) == 0:
            return [], []
        
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
