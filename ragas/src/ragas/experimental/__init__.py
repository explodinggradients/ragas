try:
    from ragas_experimental import *  # noqa: F403, F401  # type: ignore
except ImportError:
    raise ImportError(
        "ragas_experimental is required for experimental features. "
        "Install with: pip install ragas_experimental"
    )
