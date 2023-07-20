from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

PROMPT = """Given a input and prediction. Evaluate the prediction only using the given criteria. 
Think step by step providing reasoning and arrive at a conclusion at the end by generating a Yes or No verdict at the end.

input: Who was the director of Los Alamos Laboratory?
prediction: Einstein was the director of  Los Alamos Laboratory.
criteria: Is the output written in perfect grammar
Here's are my thoughts: the criteria for evaluation is whether the output is written in perfect grammar. In this case, the output is grammatically correct. Therefore, the answer is:\n\nYes

input:{}
prediction:{}
criteria:{}
Here's are my thoughts:
"""

@dataclass
class Criteria:
    name:str
    definition:str
        
    def __call__(self, question: str,answer:str, context:Optional[str]=None):
        if context is not None:
            question = f"{question } answer using context: {context}"
        return PROMPT.format(question, answer, self.definition)
    

