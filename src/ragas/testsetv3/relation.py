import typing as t
from dataclasses import dataclass
from langchain_core.documents import Document as LCDocument
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class RelationExtractor(ABC):
    name: str
    attribute1: str
    attribute2: str
    
    def get_attribute(self, doc: LCDocument, attribute: str):
        
        if hasattr(doc, self.attribute1):
            return getattr(doc, attribute)
        elif attribute in doc.metadata:
            return doc.metadata[attribute]
        else:
            return None
        
    @abstractmethod
    def extract(self, doc1: LCDocument, doc2: LCDocument) -> t.Any:
        pass
    
    
@dataclass
class Jaccardsimilarity(RelationExtractor):

    def extract(self, doc1: LCDocument, doc2: LCDocument): 
    
        a = self.get_attribute(doc1, self.attribute1)
        b = self.get_attribute(doc2, self.attribute2)
        if not isinstance(a, list) or not isinstance(b, list): 
            raise ValueError("Attributes must be lists")
        a = set(a)
        b = set(b)
        return len(a.intersection(b)) / len(a.union(b))
    

@dataclass
class EmbeddingSimilarity(RelationExtractor):
    
    def extract(self, doc1: LCDocument, doc2: LCDocument) -> t.Any:
        
        embedding1 = getattr(doc1, self.attribute1)
        embedding2 = getattr(doc2, self.attribute2)
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        
        return similarity
    
    



if __name__ == "__main__":
    
    from langchain_core.documents import Document as LCDocument
    text = """
    Contact us at info@example.com or visit https://www.example.com for more information.
    Alternatively, email support@service.com or check http://service.com.
    You can also visit our second site at www.secondary-site.org or email us at secondary-info@secondary-site.org.
    """
    
    docs = [LCDocument(page_content=text, metadata={'headlines':['one','two']})]
    
    jaccard_overlap = Jaccardsimilarity(name="jaccard", attribute1="headlines", attribute2="headlines")
    score = jaccard_overlap.extract(docs[0], docs[0])
