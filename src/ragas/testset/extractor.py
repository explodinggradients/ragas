from __future__ import annotations

import typing as t
from abc import abstractmethod, ABC
from dataclasses import dataclass

from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.testset.docstore import Node

@dataclass
class Extractor(ABC):
    llm: BaseRagasLLM
    
    @abstractmethod
    def extract(self, node: Node) -> t.Any:
        ...
        
  
      
class keyphraseExtractor(Extractor):
    
    
    def extract(self, node: Node) -> t.Any:
        self.llm.
    