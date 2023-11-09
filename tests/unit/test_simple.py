from __future__ import annotations

import typing as t


def test_import():
    import ragas
    from ragas.testset.testset_generator import TestsetGenerator

    assert TestsetGenerator is not None
    assert ragas is not None


def test_type_casting():
    t.cast(t.List[int], [1, 2, 3])


def test_import_metrics():
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy,
        faithfulness,
    )
    from ragas.metrics.critique import harmfulness

    assert harmfulness is not None
