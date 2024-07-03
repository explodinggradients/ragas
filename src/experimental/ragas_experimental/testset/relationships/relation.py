import logging
import typing as t
from dataclasses import dataclass

from ragas_experimental.testset.graph import Node, NodeLevel, Relationship, schema
from ragas_experimental.testset.relationships.base import Similarity

logger = logging.getLogger(__name__)


@dataclass
class RelationshipBuilder:
    @staticmethod
    def form_relations(
        nodes: t.List[Node],
        relationships: t.List[Relationship],
        similarity_functions: t.List[Similarity],
        node_level: NodeLevel,
        kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        kwargs = kwargs or {}
        score_threshold = kwargs.get("score_threshold", 0.0)

        query = """
            {{
            filterNodes(level: {node_level}) {{
                id
                label
                properties
                }}
            }}
                """.format(
            node_level=node_level.name
        )
        results = schema.execute(
            query, context={"nodes": nodes, "relationships": relationships}
        )

        if results.errors:
            raise Exception(results.errors)
        if not results.data.get("filterNodes"):
            logger.warning("No new relations created due to empty results.")
            return (nodes, relationships)

        node_ids = [item.get("id") for item in results.data["filterNodes"]]
        nodes_ = [node for node in nodes if node.id in node_ids]
        for extractor in similarity_functions:
            similarity_matrix = extractor.extract(nodes_, nodes_)
            for i, row in enumerate(similarity_matrix):
                new_relationships = []
                for j, score in enumerate(row):
                    if i != j and score >= score_threshold:
                        relationship = Relationship(
                            source=nodes_[i],
                            target=nodes_[j],
                            label=extractor.name,
                            properties={"score": score},
                        )
                        new_relationships.append(relationship)
                        relationship.source.relationships.append(relationship)
                        relationship.target.relationships.append(relationship)

                relationships.extend(new_relationships)

        return (nodes, relationships)
