"""Context Entity Recall prompts - V1-identical using exact PydanticPrompt.to_string() output."""

import json


def extract_entities_prompt(text: str) -> str:
    """
    V1-identical entity extraction prompt using exact PydanticPrompt.to_string() output.
    Args:
        text: The text to extract entities from
    Returns:
        V1-identical prompt string for the LLM
    """

    safe_text = json.dumps(text)

    return f"""Given a text, extract unique entities without repetition. Ensure you consider different forms or mentions of the same entity as a single entity.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"entities": {{"items": {{"type": "string"}}, "title": "Entities", "type": "array"}}}}, "required": ["entities"], "title": "EntitiesList", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.
--------EXAMPLES-----------
Example 1
Input: {{
    "text": "The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally. Millions of visitors are attracted to it each year for its breathtaking views of the city. Completed in 1889, it was constructed in time for the 1889 World's Fair."
}}
Output: {{
    "entities": [
        "Eiffel Tower",
        "Paris",
        "France",
        "1889",
        "World's Fair"
    ]
}}
Example 2
Input: {{
    "text": "The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement. Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80. It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles."
}}
Output: {{
    "entities": [
        "Colosseum",
        "Rome",
        "Flavian Amphitheatre",
        "Vespasian",
        "AD 70",
        "Titus",
        "AD 80"
    ]
}}
Example 3
Input: {{
    "text": "The Great Wall of China, stretching over 21,196 kilometers from east to west, is a marvel of ancient defensive architecture. Built to protect against invasions from the north, its construction started as early as the 7th century BC. Today, it is a UNESCO World Heritage Site and a major tourist attraction."
}}
Output: {{
    "entities": [
        "Great Wall of China",
        "21,196 kilometers",
        "7th century BC",
        "UNESCO World Heritage Site"
    ]
}}
Example 4
Input: {{
    "text": "The Apollo 11 mission, which launched on July 16, 1969, marked the first time humans landed on the Moon. Astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins made history, with Armstrong being the first man to step on the lunar surface. This event was a significant milestone in space exploration."
}}
Output: {{
    "entities": [
        "Apollo 11 mission",
        "July 16, 1969",
        "Moon",
        "Neil Armstrong",
        "Buzz Aldrin",
        "Michael Collins"
    ]
}}
-----------------------------
Now perform the same with the following input
input: {{
    "text": {safe_text}
}}
Output: """
