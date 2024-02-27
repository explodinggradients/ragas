from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field
from typing import Dict, List


from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)

CONTEXT_ENTITY_RECALL = Prompt(
    name="context_entity_recall",
    instruction="""Based on the provided contexts and the ground truth, extract entities
    present in the ground truth and the context. Make sure you do not repeat entities if they
    are present more than once, in some different form, in ground_truth or context. 
    Output should be in plain JSON format without any additional text. 
    Follow the structure provided in the examples.""",
    input_keys=["ground_truth", "context"],
    output_key="output",
    output_type="json",
    examples=[
        {
            "ground_truth": """The Eiffel Tower is located in Paris, France. 
            It is one of the most famous landmarks in the world. 
            The tower was completed in 1889.""",
            "context": """The Eiffel Tower attracts millions of visitors each year. 
            It offers breathtaking views of Paris from its top. 
            The construction of the tower was completed in time for the 1889 World's Fair.""",
            "output":{
                    "entities_in_ground_truth": ['Eiffel Tower', 'Paris', 'France', '1889'],
                    "entities_in_context": ['Eiffel Tower', 'Paris', '1889']
                    }
        },
        {
            "ground_truth": """The Colosseum, also known as the Flavian Amphitheatre, is an oval amphitheatre in the centre of Rome, Italy. 
                        It is the largest ancient amphitheatre ever built and is considered one of the greatest works of Roman architecture and engineering. 
                        The Colosseum could hold an estimated 50,000 to 80,000 spectators and was used for gladiatorial contests and public spectacles.""",
            "context": """The Colosseum is an iconic symbol of ancient Rome's power and grandeur. 
                        It is a popular tourist attraction and a testament to the architectural and engineering prowess of the Romans. 
                        The construction of the Colosseum began in AD 70 under the emperor Vespasian and was completed in AD 80 under his successor and heir Titus.""",
            "output":{
                    "entities_in_ground_truth": ['Colosseum', 'Rome', 'Italy'],
                    "entities_in_context": ['Colosseum', 'Rome', 'Titus', 'AD 70', 'Vespasian', 'Titus']
                    }
        },
        {
            "ground_truth": """The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. 
                        It was built along the northern borders of China to protect the Chinese states and empires against the raids and invasions of the various nomadic groups of the Eurasian Steppe. 
                        The wall stretches over approximately 21,196 kilometers (13,171 miles) from east to west of China.""",
            "context": """The Great Wall of China is one of the most impressive architectural feats in history. 
                        It is a UNESCO World Heritage Site and attracts millions of tourists each year. 
                        The construction of the wall began as early as the 7th century BC and continued for centuries.""",
            "output":{
                    "entities_in_ground_truth": ['Great Wall of China', 'China', 'Eurasian Steppe'],
                    "entities_in_context": ['Great Wall of China', 'UNESCO World Heritage Site', '7th century BC']
                    }
        },
        {
            "ground_truth": """The Apollo 11 mission was the first crewed mission to land humans on the Moon. 
                        It was launched by NASA on July 16, 1969, and the astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins were onboard. 
                        Neil Armstrong became the first person to step onto the lunar surface on July 20, 1969.""",
            "context": """The Apollo 11 mission was a monumental achievement in human history. 
                        It marked the culmination of years of scientific and technological advancements. 
                        The successful landing of the lunar module 'Eagle' on the Moon paved the way for future space exploration.""",
            "output":{
                    "entities_in_ground_truth": ['Apollo 11 mission', 'Moon', 'NASA', 'July 16, 1969', 'Neil Armstrong', 'Buzz Aldrin', 'Michael Collins'],
                    "entities_in_context": ['Apollo 11 mission', 'Moon', 'lunar module']
                    }
        }
    ]
)

# This function is needed for post-processing LLM reponse
# Sometimes the LLM, despite of instructing not to, adds 
# this - ```json - to the response. So this function extracts only the 
# valid part from the response
def _extract_valid_json(text):
    # Find the first occurrence of a valid JSON object
    start_index = -1
    end_index = -1
    for i in range(len(text)):
        if text[i] == '{':
            start_index = i
            break

    if start_index != -1:
        # Find the last occurrence of a valid JSON object
        for i in range(len(text) - 1, -1, -1):
            if text[i] == '}':
                end_index = i + 1
                break

    if start_index != -1 and end_index != -1:
        json_text = text[start_index:end_index]
        try:
            # Attempt to load the extracted JSON
            json_data = json.loads(json_text)
            return json_data
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return None
    else:
        print("No valid JSON found in the text.")
        return None

@dataclass
class ContextEntityRecall(MetricWithLLM): 
    """
    Calculates recall based on entities present in ground truth and context.
    Let CN be the set of entities present in context,
    GN be the set of entities present in the ground truth.

    Then we define can the context entity recall as follows:
    Context Entity recall = | CN ∩ GN | / | GN |

    If this quantity is 1, we can say that the retrieval mechanism has
    retrieved context which covers all entities present in the ground truth,
    thus being a useful retrieval. Thus this can be used to evaluate retrieval 
    mechanisms in specific use cases where entities matter, for example, a 
    tourism help chatbot.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_entity_recall" # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qc  # type: ignore
    context_entity_recall_prompt: Prompt = field(default_factory=lambda: CONTEXT_ENTITY_RECALL)
    batch_size: int = 15

    def _compute_score(self, response: str) -> float:
        response_dict = _extract_valid_json(response)
        entities_in_ground_truth = set(response_dict['entities_in_ground_truth'])
        entities_in_context = set(response_dict['entities_in_context'])
        
        num_entities_in_both = len(set(entities_in_context).intersection(set(entities_in_ground_truth)))
        num_entities_in_ground_truth = len(entities_in_ground_truth)
        return num_entities_in_both/num_entities_in_ground_truth
    
    def _score(self, row: Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not initialized"

        ground_truth, contexts = row["ground_truths"], row["contexts"]
        result = self.llm.generate_text(
            prompt=self.context_entity_recall_prompt.format(
                ground_truth="\n".join(ground_truth), context="\n".join(contexts)
            ),
            callbacks=callbacks
        )
        return self._compute_score(result.generations[0][0].text)
    
    async def _ascore(self, row: Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not initialized"

        ground_truth, contexts = row["ground_truths"], row["contexts"]
        result = self.llm.agenerate_text(
            prompt=self.context_entity_recall_prompt.format(
                ground_truth="\n".join(ground_truth), context="\n".join(contexts)
            ),
            callbacks=callbacks
        )
        return self._compute_score(result.generations[0][0].text)
    
    def save(self, cache_dir: str | None = None) -> None:
        return self.context_entity_recall_prompt.save(cache_dir)

context_entity_recall = ContextEntityRecall(batch_size=15)