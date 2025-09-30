"""
Demo showing how embedding_factory routes between legacy and modern implementations.
"""

import os
from ragas.embeddings import embedding_factory

def test_routing():
    """Demonstrate how the factory routes between different implementations."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🔄 Embedding Factory Routing Demo")
    print("=" * 50)
    
    # ===== LEGACY ROUTING (LangChain-based) =====
    print("\n📋 1. LEGACY ROUTING (triggers LangchainEmbeddingsWrapper)")
    
    legacy_examples = [
        # These trigger LEGACY because no client + looks like model name
        ("embedding_factory()", lambda: embedding_factory()),
        ("embedding_factory('text-embedding-ada-002')", lambda: embedding_factory("text-embedding-ada-002")),
        ("embedding_factory('openai')", lambda: embedding_factory("openai")),
        ("embedding_factory('gpt-3.5-turbo')", lambda: embedding_factory("gpt-3.5-turbo")),
    ]
    
    for desc, factory_call in legacy_examples:
        try:
            embedder = factory_call()
            print(f"   ✅ {desc}")
            print(f"      Type: {type(embedder).__name__}")
            print(f"      Wrapped: {type(embedder.embeddings).__name__ if hasattr(embedder, 'embeddings') else 'N/A'}")
        except Exception as e:
            print(f"   ❌ {desc}: {str(e)[:60]}...")
    
    # ===== MODERN ROUTING (Native Ragas) =====
    print("\n📋 2. MODERN ROUTING (triggers native Ragas implementations)")
    
    # These would trigger MODERN if we had the clients
    modern_examples = [
        "embedding_factory('openai', model='text-embedding-3-small', client=openai_client)",
        "embedding_factory('huggingface', model='sentence-transformers/all-MiniLM-L6-v2')",
        "embedding_factory('google', client=vertex_client)",
        "embedding_factory('litellm', model='text-embedding-ada-002', client=litellm_client)",
    ]
    
    for desc in modern_examples:
        print(f"   📝 {desc}")
        print(f"      → Would create native provider implementation")
    
    # ===== EXPLICIT INTERFACE CONTROL =====
    print("\n📋 3. EXPLICIT INTERFACE CONTROL")
    
    try:
        # Force legacy
        legacy_forced = embedding_factory("openai", interface="legacy")
        print(f"   ✅ interface='legacy': {type(legacy_forced).__name__}")
        
        # Force modern (would fail without client for providers that need it)
        print(f"   📝 interface='modern': Would use native implementations")
        
    except Exception as e:
        print(f"   ⚠️  Modern interface needs proper client: {str(e)[:60]}...")
    
    # ===== DETECTION LOGIC DEMO =====
    print("\n📋 4. DETECTION LOGIC")
    
    detection_cases = [
        ("'openai'", "openai", None, None),
        ("'text-embedding-ada-002'", "text-embedding-ada-002", None, None),
        ("'huggingface'", "huggingface", None, None),
        ("'openai' + client", "openai", None, "mock_client"),
    ]
    
    for desc, provider, model, client in detection_cases:
        # Import the detection function
        from ragas.embeddings.base import _is_legacy_embedding_call
        is_legacy = _is_legacy_embedding_call(provider, model, client, "auto")
        route = "LEGACY" if is_legacy else "MODERN"
        print(f"   {desc:<25} → {route}")
    
    print("\n" + "=" * 50)
    print("💡 Key Points:")
    print("   • Legacy: LangchainEmbeddingsWrapper (deprecated)")
    print("   • Modern: Native Ragas implementations")
    print("   • Auto-detection based on parameters")
    print("   • Legacy triggered by: no client + (model-like name OR 'openai')")
    print("   • Modern triggered by: client provided OR explicit provider names")


if __name__ == "__main__":
    test_routing()
