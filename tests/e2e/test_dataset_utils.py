"""Utilities for creating test datasets in e2e tests."""

import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Sample data structure matching the amnesty_qa dataset
SAMPLE_AMNESTY_DATA = [
    {
        "user_input": "What are the global implications of the USA Supreme Court ruling on abortion?",
        "reference": "The global implications of the USA Supreme Court ruling on abortion are significant. The ruling has led to limited or no access to abortion for one in three women and girls of reproductive age in states where abortion access is restricted. These states also have weaker maternal health support, higher maternal death rates, and higher child poverty rates. Additionally, the ruling has had an impact beyond national borders due to the USA's geopolitical and cultural influence globally.",
        "response": "The global implications of the USA Supreme Court ruling on abortion can be significant, as it sets a precedent for other countries and influences the global discourse on reproductive rights. The Supreme Court's ruling can serve as a reference point for other countries grappling with their own abortion laws.",
        "retrieved_contexts": [
            "In 2022, the USA Supreme Court handed down a decision ruling that overturned 50 years of jurisprudence recognizing a constitutional right to abortion.",
            "This decision has had a massive impact: one in three women and girls of reproductive age now live in states where abortion access is either totally or near-totally inaccessible.",
            "The USA Supreme Court ruling has also had impacts beyond national borders due to the geopolitical and cultural influence wielded by the USA globally.",
        ],
    },
    {
        "user_input": "How does climate change affect human rights?",
        "reference": "Climate change poses significant threats to human rights by affecting access to water, food security, health, and adequate housing. It disproportionately impacts vulnerable populations and can lead to displacement and migration.",
        "response": "Climate change impacts human rights through multiple pathways including threats to life, health, food, water, and adequate standard of living. The effects are often most severe for marginalized communities.",
        "retrieved_contexts": [
            "Climate change threatens the effective enjoyment of human rights including life, water and sanitation, food, health, housing, and livelihoods.",
            "The impacts of climate change will be felt most acutely by those segments of the population who are already in vulnerable situations.",
            "Climate change is already displacing people and will continue to do so in the future.",
        ],
    },
]


def load_amnesty_dataset_safe(config: str = "english_v3"):
    """
    Safely load the amnesty_qa dataset, falling back to local data if remote fails.

    Args:
        config: Dataset configuration name (e.g., "english_v3", "english_v2")

    Returns:
        Dataset: The loaded dataset
    """
    try:
        logger.info(f"Attempting to load amnesty_qa dataset with config '{config}'")
        dataset = load_dataset("explodinggradients/amnesty_qa", config)["eval"]
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load remote dataset: {e}")
        logger.info("Using local sample data as fallback")

        # Create a local dataset from sample data
        local_dataset = Dataset.from_list(SAMPLE_AMNESTY_DATA)
        logger.info(f"Created local dataset with {len(local_dataset)} samples")
        return local_dataset
