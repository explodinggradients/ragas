from __future__ import annotations

import os
import warnings
from functools import lru_cache

DEBUG_ENV_VAR = "RAGAS_DEBUG"
# constant to tell us that there is no key passed to the llm/embeddings
NO_KEY = "no-key"

# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_RAGAS_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "ragas")
RAGAS_CACHE_HOME = os.path.expanduser(os.getenv("RAGAS_HOME", DEFAULT_RAGAS_CACHE_HOME))


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
