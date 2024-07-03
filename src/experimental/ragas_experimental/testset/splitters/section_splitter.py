import re
import typing as t

import numpy as np
from langchain_core.documents import Document as LCDocument
from ragas_experimental.testset.graph import Node, NodeLevel, NodeType, Relationship
from ragas_experimental.testset.utils import merge_dicts


class HeadlineSplitter:
    def __init__(
        self,
        max_leaf_size: int = 4000,
        length_function=len,
        common_metadata_keys: t.List[str] = ["source"],
    ):
        self._length_function = length_function
        self._max_leaf_size = max_leaf_size
        self._common_metadata_keys = common_metadata_keys

    def _find_headline_indices(self, text, headlines):
        index_dict = {}
        for headline in headlines:
            # Build a regex pattern to match the headline with newlines before and after
            pattern = rf"(?<=\n){re.escape(headline)}"
            matches = re.finditer(pattern, text)
            first_match = next(matches, None)
            if first_match:
                index_dict[headline] = first_match.start()
        return index_dict

    def _split_text_by_headlines(self, text: str, headlines: t.List[str]):
        indices = []
        seperators = []
        headline_indices = self._find_headline_indices(text, headlines)
        values = list(headline_indices.values())
        keys = list(headline_indices.keys())
        values.append(len(text))
        for key, start_idx, end_idx in zip(keys, values[:-1], values[1:]):
            indices.append((start_idx, end_idx))
            seperators.append(key)

        chunks = [(text[idx[0] : idx[1]], sep) for idx, sep in zip(indices, seperators)]
        return chunks

    def _reassign_metadata(self, document: LCDocument, chunks: t.List[LCDocument]):
        extractive_metadata_keys = document.metadata.get("extractive_metadata_keys", [])
        for chunk in chunks:
            page_content = chunk.page_content
            text_chunk_metadata = {"extractive_metadata_keys": extractive_metadata_keys}
            for metadata_key in extractive_metadata_keys:
                metadata = document.metadata.get(metadata_key)
                if isinstance(metadata, str):
                    idx = page_content.find(metadata)
                    if idx != -1:
                        text_chunk_metadata[metadata_key] = metadata
                elif isinstance(metadata, list):
                    metadata_match_idx = [page_content.find(item) for item in metadata]
                    metadata_idx = [
                        idx
                        for idx, match_idx in enumerate(metadata_match_idx)
                        if match_idx != -1
                    ]
                    if metadata_idx:
                        text_chunk_metadata[metadata_key] = [
                            metadata[i] for i in metadata_idx
                        ]
            text_chunk_metadata = merge_dicts(chunk.metadata, text_chunk_metadata)
            chunk.metadata = text_chunk_metadata
            for key in self._common_metadata_keys:
                if key in document.metadata:
                    chunk.metadata[key] = document.metadata[key]
        return chunks

    def _split_document(
        self, document: LCDocument, headlines: t.List[str]
    ) -> t.List[LCDocument]:
        chunks = self._split_text_by_headlines(document.page_content, headlines)
        chunks = [
            LCDocument(page_content=chunk[0], metadata={"headline": chunk[1]})
            for chunk in chunks
        ]
        chunks = self._reassign_metadata(document, chunks)
        return chunks

    def _get_nodes_relationships(
        self,
        node: Node,
        headlines: t.List[str],
        nodes: t.List[Node],
        relationships: t.List[Relationship],
    ):
        document = LCDocument(
            page_content=node.properties["page_content"],
            metadata=node.properties["metadata"],
        )
        node_level = node.level
        chunks = self._split_document(document, headlines)
        node_idx = [idx for idx, n in enumerate(nodes) if n.id == node.id][0]
        for chunk in chunks:
            nodes.append(
                Node(
                    label=NodeType.CHUNK,
                    properties={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    relationships=[],
                    level=node_level.next_level(),
                )
            )
            relationship = Relationship(
                source=node,
                target=nodes[-1],
                label="child",
                properties={"seperator": chunk.metadata["headline"]},
            )
            relationships.append(relationship)
            nodes[node_idx].relationships.append(relationship)
            relationship = Relationship(
                source=nodes[-1],
                target=node,
                label="parent",
                properties={"seperator": chunk.metadata["headline"]},
            )
            nodes[-1].relationships.append(relationship)
            relationships.append(relationship)
        return nodes, relationships

    def split_nodes_by_attribute(self, document: LCDocument, attribute: str):
        node = Node(
            label=NodeType.DOC,
            properties={
                "page_content": document.page_content,
                "metadata": document.metadata,
            },
            relationships=[],
            level=NodeLevel.LEVEL_0,
        )
        nodes = []
        relationships = []
        nodes.append(node)
        attr_values = document.metadata.get(attribute, [])
        if isinstance(attr_values, dict):
            headlines = attr_values.keys()
            nodes, relationships = self._get_nodes_relationships(
                node, headlines, nodes, relationships
            )
            level_1_nodes = [node for node in nodes if node.level == NodeLevel.LEVEL_1]
            for node in level_1_nodes:
                subheadings = attr_values.get(
                    node.properties["metadata"]["headline"], []
                )
                if subheadings:
                    nodes, relationships = self._get_nodes_relationships(
                        node, subheadings, nodes, relationships
                    )

        nodes, relationships = self._ensure_leaf_node_size(nodes, relationships)
        return nodes, relationships

    def get_all_leaf_nodes(self, nodes: t.List[Node]) -> t.List[Node]:
        leaf_nodes = []
        for node in nodes:
            child_relationships = [
                relation
                for relation in node.relationships
                if relation.label == "child" and relation.source.id == node.id
            ]
            if not child_relationships:
                leaf_nodes.append(node)

        return leaf_nodes

    def _ensure_leaf_node_size(
        self, nodes: t.List[Node], relationships: t.List[Relationship]
    ) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        leaf_nodes = self.get_all_leaf_nodes(nodes)
        for leaf_node in leaf_nodes:
            chunks = []
            seperator = leaf_node.properties["metadata"].get("headline")
            doc = LCDocument(
                page_content=leaf_node.properties["page_content"],
                metadata=leaf_node.properties["metadata"],
            )
            page_content = leaf_node.properties["page_content"]
            content_length = self._length_function(page_content)
            if content_length > self._max_leaf_size:
                num_chunks = np.ceil(content_length / self._max_leaf_size)
                start_idx = [
                    max(0, i * self._max_leaf_size - 200)
                    for i in range(int(num_chunks))
                ]
                end_idx = [
                    (i + 1) * self._max_leaf_size for i in range(int(num_chunks))
                ]
                for start, end in zip(start_idx, end_idx):
                    chunks.append(
                        LCDocument(
                            page_content=page_content[start:end],
                            metadata={},
                        )
                    )
                chunks = self._reassign_metadata(doc, chunks)
                leaf_node_idx = [
                    idx for idx, n in enumerate(nodes) if n.id == leaf_node.id
                ][0]
                for chunk in chunks:
                    nodes.append(
                        Node(
                            label=NodeType.CHUNK,
                            properties={
                                "page_content": chunk.page_content,
                                "metadata": chunk.metadata,
                            },
                            relationships=[],
                            level=leaf_node.level.next_level(),
                        )
                    )
                    relationship = Relationship(
                        source=leaf_node,
                        target=nodes[-1],
                        label="child",
                        properties={"seperator": seperator},
                    )
                    relationships.append(relationship)
                    nodes[leaf_node_idx].relationships.append(relationship)
                    relationship = Relationship(
                        source=nodes[-1],
                        target=leaf_node,
                        label="parent",
                        properties={"seperator": seperator},
                    )
                    nodes[-1].relationships.append(relationship)
                    relationships.append(relationship)

        return nodes, relationships

    def split_documents(self, documents: t.Sequence[LCDocument], attribute: str):
        nodes = []
        relationships = []
        for doc in documents:
            doc_nodes, doc_relationships = self.split_nodes_by_attribute(doc, attribute)
            nodes.extend(doc_nodes)
            relationships.extend(doc_relationships)

        return nodes, relationships
