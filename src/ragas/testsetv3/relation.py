import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document as LCDocument


@dataclass
class Similarity(ABC):
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
    def extract(self, doc1: t.List[LCDocument], doc2: t.List[LCDocument]) -> t.Any:
        pass


@dataclass
class Jaccard(Similarity):
    def extract(self, doc1: t.List[LCDocument], doc2: t.List[LCDocument]):
        jaccard_similarity_matrix = np.zeros((len(doc1), len(doc2)))

        doc1_items = [self.get_attribute(doc, self.attribute1) for doc in doc1]
        doc2_items = [self.get_attribute(doc, self.attribute2) for doc in doc2]
        if all(isinstance(item, list) for item in doc1_items) and all(
            isinstance(item, list) for item in doc2_items
        ):
            for i, a in enumerate(doc1_items):
                for k, b in enumerate(doc2_items):
                    intersection = len(set(a).intersection(set(b)))
                    union = len(set(a).union(set(b)))
                    jaccard_similarity_matrix[i][k] = intersection / union

        return jaccard_similarity_matrix


@dataclass
class Cosine(Similarity):
    def extract(self, doc1: t.List[LCDocument], doc2: t.List[LCDocument]) -> t.Any:
        embeddings_1 = [getattr(doc, self.attribute1) for doc in doc1]
        embeddings_2 = [getattr(doc, self.attribute2) for doc in doc2]
        embeddings_1 = np.array(embeddings_1)
        embeddings_2 = np.array(embeddings_2)
        cosine_similarity_matrix = np.dot(embeddings_1, embeddings_2.T) / (
            np.linalg.norm(embeddings_1) * np.linalg.norm(embeddings_2)
        )
        return cosine_similarity_matrix


if __name__ == "__main__":
    from langchain_core.documents import Document as LCDocument

    text = """
    Contact us at info@example.com or visit https://www.example.com for more information.
    Alternatively, email support@service.com or check http://service.com.
    You can also visit our second site at www.secondary-site.org or email us at secondary-info@secondary-site.org.
    """

    docs = [LCDocument(page_content=text, metadata={"headlines": ["one", "two"]})]

    jaccard_overlap = Jaccardsimilarity(
        name="jaccard", attribute1="headlines", attribute2="headlines"
    )
    score = jaccard_overlap.extract(docs, docs)
