def test_import_module():
    import ragas.metrics
    import ragas.metrics.critique

    test_metrics = [
        "answer_correctness",
        "answer_relevancy",
        "answer_similarity",
        "context_recall",
        "context_precision",
        "faithfulness",
    ]

    test_critique = [
        "harmfulness",
        "maliciousness",
        "coherence",
        "correctness",
        "conciseness",
    ]

    assert ragas.metrics is not None, "module is not imported"

    for metric in test_metrics:
        assert hasattr(ragas.metrics, metric)

    for metric in test_critique:
        assert hasattr(ragas.metrics.critique, metric)


def test_import_in_debug_mode():
    """
    if `RAGAS_DEBUG` is set to `True`, the module should be imported with
    logging level set to `DEBUG`
    """
    import os

    os.environ["RAGAS_DEBUG"] = "True"

    from ragas.utils import get_debug_mode

    assert get_debug_mode() is True
