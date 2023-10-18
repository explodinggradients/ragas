def test_import():
    import ragas
    from ragas.testset.testset_generator import TestsetGenerator

    assert TestsetGenerator is not None
    assert ragas is not None
