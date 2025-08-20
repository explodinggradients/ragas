from __future__ import annotations

__all__ = ["SimpleExampleStore", "SimpleInMemoryExampleStore", "DynamicFewShotPrompt"]

import gzip
import json
import typing as t
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ragas.embeddings.base import BaseRagasEmbedding as BaseEmbedding

from .simple_prompt import Prompt

if t.TYPE_CHECKING:
    from pydantic import BaseModel


class SimpleExampleStore(ABC):
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


class SimpleInMemoryExampleStore(SimpleExampleStore):
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
        return self.embedding_model.embed_query(text)

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
        self.example_store = SimpleInMemoryExampleStore(embedding_model=embedding_model)
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

    def save(self, path: str, include_embeddings: bool = True) -> None:
        """
        Save the DynamicFewShotPrompt to a JSON file.

        Parameters:
        -----------
        path : str
            File path to save to. Use .gz extension for compression.
        include_embeddings : bool, default=True
            Whether to include embeddings in the saved file. If False,
            embeddings will be recomputed on load.

        Note:
        -----
        If the prompt has a response_model or embedding_model, their schemas
        will be saved for reference but the models themselves cannot be serialized.
        You'll need to provide them when loading.
        """
        if self.response_model:
            warnings.warn(
                "response_model cannot be saved and will be lost. "
                "You'll need to set it manually after loading using: "
                "DynamicFewShotPrompt.load(path, response_model=YourModel)"
            )

        if self.example_store.embedding_model:
            warnings.warn(
                "embedding_model cannot be saved and will be lost. "
                "You'll need to set it manually after loading using: "
                "DynamicFewShotPrompt.load(path, embedding_model=YourModel)"
            )

        data = {
            "format_version": "1.0",
            "type": "DynamicFewShotPrompt",
            "instruction": self.instruction,
            "examples": [
                {"input": inp, "output": out}
                for inp, out in self.example_store._examples
            ],
            "response_model_info": self._serialize_response_model_info(),
            "max_similar_examples": self.max_similar_examples,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model_info": self._serialize_embedding_model_info(),
        }

        # Optionally include embeddings
        if include_embeddings and self.example_store._embeddings_list:
            data["embeddings"] = self.example_store._embeddings_list

        file_path = Path(path)
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except (OSError, IOError) as e:
            raise ValueError(f"Cannot save DynamicFewShotPrompt to {path}: {e}")

    def _serialize_embedding_model_info(self) -> t.Optional[t.Dict]:
        """Serialize embedding model information for storage."""
        if not self.example_store.embedding_model:
            return None

        return {
            "class_name": self.example_store.embedding_model.__class__.__name__,
            "module": self.example_store.embedding_model.__class__.__module__,
            "note": "You must provide this model when loading",
        }

    @classmethod
    def load(
        cls,
        path: str,
        response_model: t.Optional["BaseModel"] = None,
        embedding_model: t.Optional[BaseEmbedding] = None,
    ) -> "DynamicFewShotPrompt":
        """
        Load a DynamicFewShotPrompt from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        embedding_model : Optional[BaseEmbedding]
            Embedding model to use for similarity calculations. Required if the
            original prompt had an embedding_model.
        response_model : Optional[BaseModel]
            Pydantic model to use for response validation. Required if the
            original prompt had a response_model.

        Returns:
        --------
        DynamicFewShotPrompt
            Loaded prompt instance

        Raises:
        -------
        ValueError
            If file cannot be loaded, is invalid, or missing required models
        """
        file_path = Path(path)

        # Load JSON data
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Cannot load DynamicFewShotPrompt from {path}: {e}")

        # Validate format
        if data.get("type") != "DynamicFewShotPrompt":
            raise ValueError(
                f"File is not a DynamicFewShotPrompt (found type: {data.get('type', 'unknown')})"
            )

        # Check if models are required but not provided
        response_model_info = data.get("response_model_info")
        if response_model_info and not response_model:
            raise ValueError(
                f"This prompt requires a response_model of type '{response_model_info['class_name']}'\\n"
                f"Usage: DynamicFewShotPrompt.load('{path}', response_model=YourModel)"
            )

        embedding_model_info = data.get("embedding_model_info")
        if embedding_model_info and not embedding_model:
            warnings.warn(
                f"This prompt was created with an embedding_model of type '{embedding_model_info['class_name']}'. "
                f"Without it, similarity-based example selection will not work. "
                f"Consider: DynamicFewShotPrompt.load('{path}', embedding_model=YourModel)"
            )

        # Extract examples
        examples = [(ex["input"], ex["output"]) for ex in data.get("examples", [])]

        # Extract DynamicFewShotPrompt-specific config
        max_similar_examples = data.get("max_similar_examples", 3)
        similarity_threshold = data.get("similarity_threshold", 0.7)

        # Create prompt instance
        prompt = cls(
            instruction=data["instruction"],
            examples=examples,
            response_model=response_model,
            embedding_model=embedding_model,
            max_similar_examples=max_similar_examples,
            similarity_threshold=similarity_threshold,
        )

        # Restore embeddings if available and compatible
        if (
            "embeddings" in data
            and embedding_model
            and len(data["embeddings"]) == len(examples)
        ):
            prompt.example_store._embeddings_list = data["embeddings"]

        # Validate response model if both provided and expected
        if response_model and response_model_info:
            prompt._validate_response_model(response_model, response_model_info)

        return prompt
