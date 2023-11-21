from __future__ import annotations

import json
import logging
import os
import warnings
from functools import lru_cache

DEBUG_ENV_VAR = "RAGAS_DEBUG"
# constant to tell us that there is no key passed to the llm/embeddings
NO_KEY = "no-key"


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        logging.basicConfig(level=logging.DEBUG)
        return True
    else:
        return False


def load_as_json(text):
    """
    validate and return given text as json
    """

    try:
        return json.loads(text)
    except ValueError:
        print(text)
        warnings.warn("Invalid json")

    return {}
