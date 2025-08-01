from __future__ import annotations

__all__ = ["ExampleStore", "InMemoryExampleStore", "DynamicFewShotPrompt"]

import typing as t
from abc import ABC, abstractmethod

import numpy as np


from ..embeddings import BaseEmbedding
from .base import Prompt

if t.TYPE_CHECKING:
    from pydantic import BaseModel


class ExampleStore(ABC):
    @abstractmethod
    def get_examples(
        self, data: t.Dict, top_k: int = 5
    ) -> t.List[t.Tuple[t.Dict, t.Dict]]:
        """Get top_k most similar examples to data."""
        pass

    @abstractmethod
    def add_example(self, input: t.Dict, output: t.Dict) -> None:
        """Add an example to the store."""
        pass


class InMemoryExampleStore(ExampleStore):
    def __init__(self, embedding_model=None):
        """
        Initialize an in-memory example store with optional embedding model.

        Args:
            embedding_model: Model used to generate embeddings (OpenAI or similar)
        """
        self.embedding_model = embedding_model
        self._examples: t.List[t.Tuple[t.Dict, t.Dict]] = []
        self._embeddings_list: t.List[t.List[float]] = []

    def _get_embedding(self, data: t.Dict) -> t.List[float]:
        """Convert input dict to an embedding vector."""
        if self.embedding_model is None:
            return []

        # Serialize the dictionary to text
        text = "\n".join([f"{k}: {v}" for k, v in data.items()])
        return self.embedding_model.embed_text(text)

    def add_example(self, input: t.Dict, output: t.Dict) -> None:
        """Add an example to the store with its embedding."""
        if not isinstance(input, dict):
            raise TypeError(f"Expected inputs to be dict, got {type(input).__name__}")
        if not isinstance(output, dict):
            raise TypeError(f"Expected output to be dict, got {type(output).__name__}")

        self._examples.append((input, output))

        if self.embedding_model:
            embedding = self._get_embedding(input)
            self._embeddings_list.append(embedding)

    def get_examples(
        self, data: t.Dict, top_k: int = 5, threshold: float = 0.7
    ) -> t.List[t.Tuple[t.Dict, t.Dict]]:
        """Get examples most similar to the input data."""
        if not self._examples:
            return []

        if not self.embedding_model or not self._embeddings_list:
            # If no embedding model, return the most recent examples
            return self._examples[-top_k:]

        # Get embedding for the query
        query_embedding = self._get_embedding(data)

        # Find most similar examples
        indices = self._get_nearest_examples(
            query_embedding, self._embeddings_list, top_k, threshold
        )

        # Return the examples at those indices
        return [self._examples[i] for i in indices]

    def _get_nearest_examples(
        self,
        query_embedding: t.List[float],
        embeddings: t.List[t.List[float]],
        top_k: int = 3,
        threshold: float = 0.7,
    ) -> t.List[int]:
        """Find indices of the nearest examples based on cosine similarity."""
        # Convert to numpy arrays for efficient computation
        query = np.array(query_embedding)
        embed_matrix = np.array(embeddings)

        # Calculate cosine similarity
        similarities = np.dot(embed_matrix, query) / (
            np.linalg.norm(embed_matrix, axis=1) * np.linalg.norm(query) + 1e-8
        )

        # Get indices of similarities above threshold
        valid_indices = np.where(similarities >= threshold)[0]

        # Sort by similarity and get top-k
        if len(valid_indices) > 0:
            top_indices = valid_indices[
                np.argsort(similarities[valid_indices])[-top_k:]
            ]
            # Convert numpy indices to Python ints
            return [int(idx) for idx in top_indices]

        # If no examples meet threshold, return most recent examples
        return list(range(max(0, len(embeddings) - top_k), len(embeddings)))

    def __len__(self):
        return len(self._examples)


class DynamicFewShotPrompt(Prompt):
    def __init__(
        self,
        instruction: str,
        examples: t.Optional[t.List[t.Tuple[t.Dict, t.Dict]]] = None,
        response_model: t.Optional[BaseModel] = None,
        embedding_model: t.Optional[BaseEmbedding] = None,
        max_similar_examples: int = 3,
        similarity_threshold: float = 0.7,
    ):
        """
        Create a dynamic few-shot prompt that selects relevant examples based on similarity.

        Parameters:
        -----------
        instruction : str
            The prompt instruction template with placeholders like {response}, {expected_answer}
        examples : Optional[List[Tuple[Dict, Dict]]]
            List of (input_dict, output_dict) pairs for few-shot learning
        response_model: Optional[BaseModel]
            The expected response model
        embedding_model : Optional[BaseEmbedding]
            Embedding model for similarity calculations. If None, falls back to recency-based selection.
        max_similar_examples : int, default=3
            Maximum number of similar examples to include in the formatted prompt
        similarity_threshold : float, default=0.7
            Minimum cosine similarity threshold (0.0-1.0) for including examples.
            Only examples with similarity >= threshold will be considered.
        """
        # Create example store first (needed for add_example override)
        self.example_store = InMemoryExampleStore(embedding_model=embedding_model)
        self.max_similar_examples = max_similar_examples
        self.similarity_threshold = similarity_threshold

        # Call parent constructor with empty examples to avoid calling add_example during init
        super().__init__(instruction, [], response_model)

        # Add examples to the store manually
        if examples:
            for input_dict, output_dict in examples:
                self.example_store.add_example(input_dict, output_dict)

    def format(self, **kwargs) -> str:
        """Format the prompt with dynamically retrieved examples."""
        prompt_parts = []

        # Add instruction with variables filled in
        prompt_parts.append(self.instruction.format(**kwargs))

        # Get dynamic examples if we have a store and inputs
        dynamic_examples = []
        if self.example_store and kwargs:
            dynamic_examples = self.example_store.get_examples(
                kwargs, self.max_similar_examples, self.similarity_threshold
            )

        # Add examples in a simple format
        if dynamic_examples:
            prompt_parts.append("Examples:")
            for i, (inputs, output) in enumerate(dynamic_examples, 1):
                example_input = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
                example_output = "\n".join([f"{k}: {v}" for k, v in output.items()])

                prompt_parts.append(
                    f"Example {i}:\nInput:\n{example_input}\nOutput:\n{example_output}"
                )

        # Combine all parts
        return "\n\n".join(prompt_parts)

    def add_example(self, input: t.Dict, output: t.Dict) -> None:
        """
        Add an example to both the prompt and the example store.

        Parameters:
        -----------
        input : Dict
            Dictionary of input values
        output : Dict
            Dictionary of output values

        Raises:
        -------
        TypeError
            If input or output is not a dictionary
        """
        # Add to example store
        if (input, output) not in self.example_store._examples:
            self.example_store.add_example(input, output)

    @classmethod
    def from_prompt(
        cls,
        prompt: Prompt,
        embedding_model: BaseEmbedding,
        max_similar_examples: int = 3,
        similarity_threshold: float = 0.7,
    ) -> "DynamicFewShotPrompt":
        """
        Create a DynamicFewShotPrompt from a Prompt object.

        Parameters:
        -----------
        prompt : Prompt
            Base prompt to convert to dynamic few-shot
        embedding_model : BaseEmbedding
            Embedding model for similarity calculations
        max_similar_examples : int, default=3
            Maximum number of similar examples to retrieve
        similarity_threshold : float, default=0.7
            Minimum similarity threshold for including examples (0.0-1.0)

        Returns:
        --------
        DynamicFewShotPrompt
            Configured dynamic few-shot prompt instance
        """
        return cls(
            instruction=prompt.instruction,
            examples=prompt.examples,
            response_model=prompt.response_model,
            embedding_model=embedding_model,
            max_similar_examples=max_similar_examples,
            similarity_threshold=similarity_threshold,
        )

    def __str__(self) -> str:
        """String representation showing the dynamic few-shot prompt configuration."""
        return (
            f"DynamicFewShotPrompt("
            f"instruction='{self.instruction}', "
            f"max_similar_examples={self.max_similar_examples}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"example_store_size={len(self.example_store)})"
        )

    __repr__ = __str__
