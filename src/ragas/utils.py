from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache

DEBUG_ENV_VAR = "RAGAS_DEBUG"
# constant to tell us that there is no key passed to the llm/embeddings
NO_KEY = "no-key"
NO_BASE_URL = "no-base-url"


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        return True
    else:
        return False


def load_as_json(text):
    """
    validate and return given text as json
    """

    try:
        return json.loads(text)
    except ValueError as e:
        warnings.warn(f"Invalid json: {e}")

    return {}
