"""
Simple example showing the progression from local to Notion backend.

This demonstrates the recommended approach:
1. Start with local backend (works immediately) 
2. Develop your evaluation workflow
3. Migrate to Notion when ready for collaboration
4. Same API - just change the backend parameter!
"""

import os
import tempfile
from pydantic import BaseModel, field_validator


class EvaluationRecord(BaseModel):
    """Simple evaluation record model."""
    question: str
    answer: str
    score: float
    
    @field_validator('score')
    @classmethod
    def score_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


def main():
    """Show the local-first approach."""
    print("🚀 Ragas Backend: Local-First Approach")
    print("=" * 50)
    
    # Step 1: Start with local backend (always works)
    print("\n📁 Step 1: Local Backend (Recommended Start)")
    try:
        from ragas_experimental.backends import LocalJSONLBackend
        
        # Create temp directory for demo
        temp_dir = tempfile.mkdtemp()
        
        # Create local backend - works immediately!
        local_backend = LocalJSONLBackend(root_dir=temp_dir)
        
        # Sample data
        sample_data = [
            {"question": "What is 2+2?", "answer": "4", "score": 1.0},
            {"question": "Capital of France?", "answer": "Paris", "score": 1.0},
        ]
        
        # Save data locally
        local_backend.save_dataset("my_evaluation", sample_data)
        print(f"✅ Saved {len(sample_data)} records locally")
        print(f"📂 Data stored in: {temp_dir}")
        
        # Load data back
        loaded_data = local_backend.load_dataset("my_evaluation")
        print(f"✅ Loaded {len(loaded_data)} records")
        
    except Exception as e:
        print(f"❌ Error: Failed to create local backend: {e}")
        return
    
    # Step 2: Show migration path to Notion (if configured)
    print("\n🌐 Step 2: Migration to Notion (When Ready)")
    
    if not (os.getenv("NOTION_TOKEN") and os.getenv("NOTION_DATABASE_ID")):
        print("⚠️  Notion not configured - showing setup info")
        print()
        print("To migrate to Notion:")
        print("1. Create Notion integration and database")
        print("2. Set environment variables:")
        print("   export NOTION_TOKEN='secret_your_token'")
        print("   export NOTION_DATABASE_ID='your_database_id'")
        print("3. Install: pip install ragas_experimental[notion]")
        print("4. Use NotionBackend instead of LocalJSONLBackend")
        print()
        print("💡 Same API - just different backend class!")
        return
    
    # If Notion is configured, show migration
    try:
        from ragas_experimental.backends import NotionBackend
        
        # Create Notion backend
        notion_backend = NotionBackend()
        
        # Save same data to Notion
        notion_backend.save_dataset("my_evaluation_shared", sample_data)
        print("✅ Migrated to Notion successfully!")
        
        # Load data back from Notion
        notion_data = notion_backend.load_dataset("my_evaluation_shared")
        print(f"✅ Loaded {len(notion_data)} records from Notion")
        print("🎉 Data now available for team collaboration")
        
    except ImportError:
        print("⚠️  Install Notion support: pip install ragas_experimental[notion]")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
    
    print("\n🎯 Key Benefits:")
    print("• Local backend: Works immediately, fast, version control friendly")
    print("• Notion backend: Team collaboration, rich UI, automatic sync")
    print("• Same API: Easy migration when you're ready")


if __name__ == "__main__":
    main()
