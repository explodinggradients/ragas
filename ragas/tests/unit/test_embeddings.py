from __future__ import annotations


def test_basic_legacy_imports():
    """Test that basic legacy imports work."""
    from ragas.embeddings import BaseRagasEmbeddings, embedding_factory

    assert BaseRagasEmbeddings is not None
    assert embedding_factory is not None


def test_debug_base_module():
    """Debug what's available in base module."""
    import ragas.embeddings.base as base_module

    # Check if RagasBaseEmbedding is in the module
    has_class = hasattr(base_module, "RagasBaseEmbedding")
    print(f"base_module has RagasBaseEmbedding: {has_class}")

    if has_class:
        cls = getattr(base_module, "RagasBaseEmbedding")
        print(f"RagasBaseEmbedding type: {type(cls)}")
        assert cls is not None
    else:
        # List what is available
        attrs = [attr for attr in dir(base_module) if not attr.startswith("_")]
        print(f"Available attributes: {attrs}")
        raise AssertionError("RagasBaseEmbedding not found in base module")


def test_direct_import_from_base():
    """Test direct import from base module."""
    try:
        from ragas.embeddings.base import RagasBaseEmbedding

        print(f"Successfully imported RagasBaseEmbedding: {RagasBaseEmbedding}")
        assert RagasBaseEmbedding is not None
    except ImportError as e:
        print(f"Import error: {e}")
        # Try to import the whole module first
        import ragas.embeddings.base

        print(f"Module imported successfully: {ragas.embeddings.base}")
        # Now try to get the class
        if hasattr(ragas.embeddings.base, "RagasBaseEmbedding"):
            cls = getattr(ragas.embeddings.base, "RagasBaseEmbedding")
            print(f"Found class via getattr: {cls}")
        else:
            print("Class not found via getattr either")
        raise


def test_main_module_import():
    """Test import from main embeddings module."""
    try:
        from ragas.embeddings import RagasBaseEmbedding

        print(f"Successfully imported from main module: {RagasBaseEmbedding}")
        assert RagasBaseEmbedding is not None
    except ImportError as e:
        print(f"Main module import error: {e}")
        # Check what's in the main module
        import ragas.embeddings

        attrs = [
            attr for attr in dir(ragas.embeddings) if "Ragas" in attr or "Base" in attr
        ]
        print(f"Ragas/Base related attributes in main module: {attrs}")
        raise
