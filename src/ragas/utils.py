from __future__ import annotations

import os
from functools import lru_cache

DEBUG_ENV_VAR = "RAGAS_DEBUG"
# constant to tell us that there is no key passed to the llm/embeddings
NO_KEY = "no-key"


@lru_cache(maxsize=1)
def get_cache_dir() -> str:
    "get cache location"
    DEFAULT_XDG_CACHE_HOME = "~/.cache"
    xdg_cache = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
    default_ragas_cache = os.path.join(xdg_cache, "ragas")
    return os.path.expanduser(os.getenv("RAGAS_CACHE_HOME", default_ragas_cache))


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        return True
    else:
        return False
