from graphene import ObjectType, String, List, Field, JSONString, Enum, Schema, Argument
import uuid

class NodeType(Enum):
    DOC = "doc"
    CHUNK = "chunk"


class Relationship(ObjectType):
    """Represents a directed relationship between two nodes in a graph.

    Attributes:
        source (Node): The source node of the relationship.
        target (Node): The target node of the relationship.
        label (str): The label of the relationship.
        properties (dict): Additional properties associated with the relationship.
    """

    source = Field(lambda: Node)
    target = Field(lambda: Node)
    label = String()
    properties = Field(JSONString)


class Node(ObjectType):
    """Represents a node in a graph with associated properties.

    Attributes:
        id (Union[str, int]): A unique identifier for the node.
        label (NodeType): The label of the node.
        properties (dict): Additional properties and metadata associated with the node.
    """
    id = String(required=True)
    label = Field(NodeType)
    properties = Field(JSONString)
    relationships = List(lambda: Relationship, label=Argument(String), property_key=Argument(String), property_value=Argument(String))

    def __init__(self, id=None, **kwargs):
        if id is None:
            id = str(uuid.uuid4())
        super().__init__(id=id, **kwargs)

    def resolve_relationships(self, info, label=None, property_key=None, property_value=None):
        relationships = info.context.get('relationships', [])
        filtered_relationships = [rel for rel in relationships if rel.source.id == self.id]
        
        if label:
            filtered_relationships = [rel for rel in filtered_relationships if rel.label == label]

        if property_key and property_value:
            filtered_relationships = [
                rel for rel in filtered_relationships 
                if rel.properties and property_key in rel.properties and rel.properties[property_key] == property_value
            ]

        return filtered_relationships


class Query(ObjectType):
    filter_nodes = List(Node, id=Argument(String), label=Argument(NodeType), property_key=Argument(String), property_value=Argument(String))

    def resolve_filter_nodes(parent, info, id=None, label=None, property_key=None, property_value=None):
        nodes = info.context.get('nodes', [])
        filtered_nodes = nodes
        
        if id:
            filtered_nodes = [node for node in nodes if node.id == id]
        
        if label:
            filtered_nodes = [node for node in nodes if node.label == label]
        
        if property_key and property_value:
            filtered_nodes = [node for node in filtered_nodes if node.properties and node.properties.get(property_key) == property_value]
        
        return filtered_nodes
    
    
schema = Schema(query=Query)