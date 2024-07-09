CLUSTER_OF_RELATED_NODES_QUERY = """
            {{
            filterNodes(label: DOC) {{
                id
                label
                properties
                relationships(label: "{label}", propertyKey: "{property}", propertyValue: "{value}", comparison: "{comparison}", targetFilter: {{label: DOC}} ) {{
                label
                properties
                target {{
                    id
                    label
                    properties
                }}
                }}
            }}
            }}
            """

LEAF_NODE_QUERY = """
{{
leafNodes(id: {id}){{
id
label
level
properties
}}
}}
"""

CHILD_NODES_QUERY = """
        {{
        filterNodes(label: DOC, level : LEVEL_0) {{
            id
            label
            properties
            relationships(label: "child") {{
            label
            properties
            target {{
                id
                label
                properties
            }}
            source {{
                id
            }}
            }}
        }}
        }}
        """
