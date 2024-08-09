import json

import numpy as np
from ragas_experimental.testset.graph import Node, NodeLevel, NodeType, Relationship

MODEL_MAX_LENGTHS = {
    "gpt-3.5-turbo-": 16385,
}


rng = np.random.default_rng(seed=42)


def merge_dicts(*dicts):
    merged_dict = {}

    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                if isinstance(value, list) and isinstance(merged_dict[key], list):
                    merged_dict[key].extend(value)
                elif isinstance(value, str) and isinstance(merged_dict[key], str):
                    merged_dict[key] += "-" + value
                elif isinstance(value, dict) and isinstance(merged_dict[key], dict):
                    merged_dict[key] = merge_dicts(merged_dict[key], value)
                else:
                    raise TypeError("Inconsistent value types for key: {}".format(key))
            else:
                merged_dict[key] = value

    return merged_dict


class GraphConverter:
    @staticmethod
    def dict_to_node(node_dict):
        relationships = [
            GraphConverter.dict_to_relationship(rel_dict, node_dict["id"])
            for rel_dict in node_dict.get("relationships", [])
        ]
        return Node(
            id=node_dict["id"],
            label=NodeType[node_dict["label"]],
            properties=json.loads(node_dict["properties"]),
            relationships=relationships,
            level=NodeLevel[node_dict["level"]] if "level" in node_dict else None,
        )

    @staticmethod
    def dict_to_relationship(rel_dict, source_id=None):
        target_node = Node(
            id=rel_dict["target"]["id"],
            label=NodeType[rel_dict["target"]["label"]],
            properties=json.loads(rel_dict["target"]["properties"]),
        )
        return Relationship(
            source=Node(id=source_id) if source_id else None,
            target=target_node,
            label=rel_dict["label"],
            properties=json.loads(rel_dict["properties"]),
        )

    @staticmethod
    def convert_dict_to_object(item):
        if "source" in item and "target" in item:
            return GraphConverter.dict_to_relationship(item)
        elif "id" in item and "label" in item:
            return GraphConverter.dict_to_node(item)
        else:
            raise ValueError("Unknown item type")

    @staticmethod
    def convert(data):
        if isinstance(data, dict):
            return GraphConverter.convert_dict_to_object(data)
        elif isinstance(data, list):
            return [GraphConverter.convert_dict_to_object(item) for item in data]
        else:
            raise ValueError("Unknown data type")
