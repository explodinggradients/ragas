"""
Comprehensive Notion backend example for Ragas experimental.

Shows complete workflow: local development → Notion collaboration.
For a simpler introduction, see: notion_backend_simple.py

This demonstrates:
- Local-first development approach
- Notion setup validation and testing  
- Migration strategies
- Error handling and best practices
"""

import os
import time
from typing import List, Optional

from pydantic import BaseModel, field_validator


class EvaluationRecord(BaseModel):
    """Model for evaluation results."""
    question: str
    answer: str
    context: str
    ground_truth: str
    score: float
    metadata: Optional[dict] = None
    
    @field_validator('score')
    @classmethod
    def score_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


def example_local_development():
    """Start with local backend - the recommended approach."""
    print("📁 Local Development (Recommended Starting Point)")
    print("-" * 50)
    
    try:
        from ragas_experimental.backends import LocalJSONLBackend
        
        # Create local backend - works immediately!
        print("Creating local backend...")
        backend = LocalJSONLBackend(root_dir="./data")
        
        # Add development data
        dev_records = [
            EvaluationRecord(
                question="What is the capital of France?",
                answer="Paris",
                context="France is a European country.",
                ground_truth="Paris",
                score=1.0,
                metadata={"difficulty": "easy"}
            ),
            EvaluationRecord(
                question="What is 2+2?",
                answer="4", 
                context="Basic arithmetic.",
                ground_truth="4",
                score=1.0,
                metadata={"difficulty": "easy"}
            ),
            EvaluationRecord(
                question="Explain quantum computing",
                answer="Quantum computing uses quantum bits...",
                context="Advanced physics topic.",
                ground_truth="Complex quantum mechanics explanation",
                score=0.7,
                metadata={"difficulty": "hard"}
            )
        ]
        
        # Convert to dictionaries for backend
        data = [record.model_dump() for record in dev_records]
        backend.save_dataset("qa_evaluation_dev", data)
        
        print(f"✅ Saved {len(data)} records to local file")
        print(f"📂 Location: ./data/qa_evaluation_dev.jsonl")
        
        # Verify local loading
        loaded_data = backend.load_dataset("qa_evaluation_dev")
        print(f"✅ Verified: loaded {len(loaded_data)} records from disk")
        
        return backend
        
    except Exception as e:
        print(f"❌ Local development failed: {e}")
        return None


def validate_notion_setup():
    """Check if Notion backend is properly configured."""
    print("\n🔍 Notion Setup Validation")
    print("-" * 30)
    
    # Check environment variables
    token = os.getenv("NOTION_TOKEN")
    database_id = os.getenv("NOTION_DATABASE_ID")
    
    if not token:
        print("❌ NOTION_TOKEN environment variable not set")
        return False
    if not database_id:
        print("❌ NOTION_DATABASE_ID environment variable not set")
        return False
        
    print(f"✅ Environment variables configured")
    
    # Test Notion backend import and connection
    try:
        from ragas_experimental.backends import NotionBackend
        
        print("Testing Notion connection...")
        backend = NotionBackend()
        
        # Test with minimal data
        test_data = [{"question": "test", "answer": "test", "score": 1.0}]
        backend.save_dataset("validation_test", test_data)
        loaded = backend.load_dataset("validation_test")
        
        print(f"✅ Notion backend operational ({len(loaded)} test records)")
        return True
        
    except ImportError:
        print("❌ Notion dependencies not installed")
        print("   Fix: pip install ragas_experimental[notion]")
        return False
    except Exception as e:
        print(f"❌ Notion connection failed: {e}")
        print("   Check token, database ID, and database schema")
        return False
def example_notion_migration():
    """Migrate from local to Notion backend."""
    print("\n🌐 Local → Notion Migration")
    print("-" * 30)
    
    # Get local data first
    local_backend = example_local_development()
    if not local_backend:
        print("❌ Cannot migrate without local data")
        return False
    
    if not validate_notion_setup():
        print("❌ Cannot migrate - Notion not properly configured")
        return False
    
    try:
        from ragas_experimental.backends import NotionBackend
        
        print("Creating Notion backend...")
        notion_backend = NotionBackend()
        
        # Load data from local backend
        local_data = local_backend.load_dataset("qa_evaluation_dev")
        
        print(f"Migrating {len(local_data)} records to Notion...")
        notion_backend.save_dataset("qa_evaluation_shared", local_data)
        
        # Verify migration
        notion_data = notion_backend.load_dataset("qa_evaluation_shared")
        print(f"✅ Migration successful! {len(notion_data)} records in Notion")
        print("🎉 Team can now collaborate via Notion interface")
        
        return notion_backend
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return None


def example_backend_comparison():
    """Compare backend options and use cases."""
    print("\n📊 Backend Comparison")
    print("-" * 20)
    
    comparison = {
        "local/jsonl": {
            "Setup": "None",
            "Speed": "Fast",
            "Collaboration": "File sharing",
            "Best for": "Development, CI/CD, version control"
        },
        "local/csv": {
            "Setup": "None",
            "Speed": "Fast", 
            "Collaboration": "Excel/Sheets",
            "Best for": "Simple data, spreadsheet analysis"
        },
        "notion": {
            "Setup": "API integration",
            "Speed": "Moderate",
            "Collaboration": "Native Notion",
            "Best for": "Team workflows, rich documentation"
        }
    }
    
    for backend, features in comparison.items():
        print(f"\n{backend}:")
        for feature, value in features.items():
            print(f"  • {feature}: {value}")


def example_advanced_patterns():
    """Show advanced usage patterns."""
    print("\n🚀 Advanced Patterns")
    print("-" * 20)
    
    patterns = [
        "✅ Local development → Notion production",
        "✅ Multi-environment: dev/staging/prod datasets", 
        "✅ Automatic backup: Notion → local files",
        "✅ Hybrid: Local for speed + Notion for sharing",
        "✅ Team workflow: Individual local → shared Notion"
    ]
    
    for pattern in patterns:
        print(f"  {pattern}")
    
    print("\n💡 Pro Tips:")
    print("  • Keep local backup even when using Notion")
    print("  • Use descriptive dataset names for team clarity")
    print("  • Notion databases can be filtered/sorted by team members")
    print("  • Migration is reversible - data isn't locked in")


def show_setup_help():
    """Provide setup guidance for new users."""
    print("\n📚 Setup Help")
    print("-" * 15)
    
    print("🎯 Recommended path:")
    print("1. Start with local backend (works immediately)")
    print("2. Develop evaluation workflow locally")
    print("3. When ready for team collaboration:")
    print("   a. Create Notion integration + database")
    print("   b. Set environment variables")
    print("   c. Install: pip install ragas_experimental[notion]")
    print("   d. Migrate data with same API")
    
    print("\n📖 Detailed guide:")
    print("  • docs/notion_backend.md - Complete setup and usage guide")


def main():
    """Run the comprehensive example."""
    print("🚀 Ragas Notion Backend: Complete Example")
    print("=" * 50)
    
    # Always start with local development
    local_dataset = example_local_development()
    
    # Show backend options
    example_backend_comparison()
    
    # Check Notion readiness
    if validate_notion_setup():
        # If configured, demonstrate migration
        notion_dataset = example_notion_migration()
        if notion_dataset:
            example_advanced_patterns()
    else:
        # If not configured, show setup help
        show_setup_help()
    
    print("\n🎉 Example completed!")
    print("\n💡 Key takeaways:")
    print("  • Local-first approach reduces setup friction")
    print("  • Same API works with any backend")
    print("  • Migration is data copying between backends")
    print("  • Choose backend based on collaboration needs")


if __name__ == "__main__":
    main()
