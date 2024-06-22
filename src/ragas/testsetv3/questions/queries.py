

ABSTRACT_QUERY =  """
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