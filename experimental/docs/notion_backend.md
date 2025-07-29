# Notion Backend for Ragas Experimental

The Notion backend lets you store evaluation datasets in Notion databases for team collaboration, while keeping the same simple API you're used to.

## 🚀 Quick Start (Recommended)

**New to Ragas?** Start with local files - no setup needed:

```python
from ragas_experimental.backends import LocalJSONLBackend

# Works immediately, no configuration required
backend = LocalJSONLBackend(root_dir="./data")
data = [{"question": "What is AI?", "answer": "Artificial Intelligence", "score": 0.95}]
backend.save_dataset("my_evaluation", data)
```

**Ready for team collaboration?** Follow the Notion setup below - takes 2 minutes.

## 📋 Notion Setup (When You're Ready)

### 1. Install Dependencies

```bash
pip install ragas_experimental[notion]
```

### 2. Connect to Notion

#### Recommended: Simple Setup

```bash
# Set these two environment variables
export NOTION_TOKEN="secret_your_token_here"
export NOTION_DATABASE_ID="your_database_id_here"
```

Then just:

```python
from ragas_experimental.backends import NotionBackend

# Option 1: Use environment variables (recommended)
import os
os.environ["NOTION_TOKEN"] = "secret_..."
backend = NotionBackend(database_id="your_database_id_here")

# Option 2: Pass directly (not recommended for production)
backend = NotionBackend(
    token="secret_your_token_here",
    database_id="your_database_id_here"
)

print("✅ Connected to Notion!")
```

**How to get these values:**

1. Go to [Notion Developers](https://developers.notion.com/docs/getting-started)  
2. Create integration → Copy token (starts with `secret_` or `ntn_`)
3. Create database → Share with integration → Copy database ID from URL

### 3. Create Database

1. In Notion, create a new page
2. Add a database with these **exact** properties:

| Property Name | Type | Required |
|---------------|------|----------|
| Name | Title | ✅ |
| Type | Select | ✅ |
| Item_Name | Text | ✅ |
| Data | Text | ✅ |
| Created_At | Date | ✅ |
| Updated_At | Date | ✅ |

For "Type" property: Add options `dataset` and `experiment`

#### 🔧 How to Add/Modify Properties

**To add a new property:**

1. Click the **"+"** button at the end of your database columns
2. Choose the property type and name it exactly as shown above

**To change an existing property type:**

1. Click on the **column header** (property name)
2. Select **"Edit property"**
3. Change the **"Type"** dropdown to the correct type
4. For "Type" property: After changing to "Select", add options: `dataset` and `experiment`

**Common fixes needed:**

- Change `Type` from "Rich Text" → **"Select"** (then add `dataset` and `experiment` options)
- Change `Created_At` from "Rich Text" → **"Date"**
- Change `Updated_At` from "Rich Text" → **"Date"**

### 4. Connect Integration

1. In your database: Click ⋯ → "Add connections"
2. Select your integration
3. Copy database ID from URL (the part right after `notion.so/`, NOT the part after `?v=`)

### 5. Set Environment Variables

```bash
export NOTION_TOKEN="secret_your_token_here"
export NOTION_DATABASE_ID="your_database_id_here"
```

### 6. Test Setup

```python
from ragas_experimental.backends import NotionBackend

backend = NotionBackend()
backend.save_dataset("test", [{"question": "Setup works?", "answer": "Yes!", "score": 1.0}])
print("✅ Notion backend ready!")
```

## 💻 Usage

### Same API, Different Backend

```python
from ragas_experimental.dataset import Dataset
from pydantic import BaseModel

class EvaluationRecord(BaseModel):
    question: str
    answer: str
    score: float

# Local backend (immediate)
local_dataset = Dataset("my_eval", backend="local/jsonl", data_model=EvaluationRecord)

# Notion backend (after setup)
notion_dataset = Dataset("my_eval", backend="notion", data_model=EvaluationRecord)

# Same operations work with both
record = EvaluationRecord(question="What is ML?", answer="Machine Learning", score=0.9)
dataset.append(record)
dataset.save()
```

### Direct Backend Usage

```python
from ragas_experimental.backends import NotionBackend

backend = NotionBackend()

# Save data
data = [
    {"question": "What is AI?", "answer": "Artificial Intelligence", "score": 0.95},
    {"question": "What is ML?", "answer": "Machine Learning", "score": 0.88}
]
backend.save_dataset("my_dataset", data)

# Load data
loaded = backend.load_dataset("my_dataset")

# List available datasets
datasets = backend.list_datasets()
```

## 🔄 Migration: Local → Notion

Already using local files? Easy migration:

```python
# Load from local
local_dataset = Dataset("my_data", backend="local/jsonl")
local_dataset.load()

# Save to Notion
notion_dataset = Dataset("my_data", backend="notion")
for item in local_dataset:
    notion_dataset.append(item)
notion_dataset.save()
```

## 📊 When to Use Each Backend

| Feature | Local Files | Notion |
|---------|-------------|--------|
| Setup time | 0 seconds | 2 minutes |
| Speed | Fast | Moderate |
| Team collaboration | File sharing | Native UI |
| Best for | Development, CI/CD | Team workflows |
| Version control | ✅ Git-friendly | ❌ API-based |
| Offline access | ✅ Always | ❌ Requires internet |

**Recommendation:** Start with local files, migrate to Notion when you need team collaboration.

## ⚙️ Configuration

### Environment Variables

- `NOTION_TOKEN`: Your integration token
- `NOTION_DATABASE_ID`: Your database ID

### Constructor Parameters

```python
# Use environment variables
backend = NotionBackend()

# Or specify directly
backend = NotionBackend(
    token="secret_your_token",
    database_id="your_database_id"
)
```

## 🚨 Troubleshooting

### Common Issues

## Troubleshooting

### "Cannot access Notion database"

- Check integration token starts with `secret_` or `ntn_`
- **Most common issue:** Database is not shared with your integration
  - Go to your database in Notion
  - Click ⋯ (three dots) → "Add connections"
  - Select your integration from the list
- Verify database ID is 32 characters (UUID format)
- Ensure your integration has the correct permissions

### "Database missing required properties"

**Error message example:**

```text
Database missing required properties: Type (expected select, found rich_text), 
Created_At (expected date, found rich_text), Updated_At (expected date, found rich_text)
```

**Solution:**

1. Go to your Notion database
2. Fix each property type mentioned in the error:
   - Click on the column header (property name)
   - Select "Edit property"
   - Change the type as needed:
     - `Type`: Change to **"Select"** → Add options: `dataset` and `experiment`
     - `Created_At`: Change to **"Date"**
     - `Updated_At`: Change to **"Date"**
3. Property names must match exactly (case-sensitive)
4. Test again after fixing all properties

### "Notion backend requires additional dependencies"

- Run: `pip install ragas_experimental[notion]`

### Rate limit errors

- Notion allows 3 requests/second
- Backend includes automatic retry with backoff
- For large datasets, consider batching operations

### Getting Help

1. Double-check database properties and names
2. Verify integration permissions
3. Test with a simple example first

## ⚡ Performance & Limits

### Notion API Limits

- **Rate limit:** 3 requests per second
- **Property size:** ~2000 characters for rich text
- **Practical limit:** Best for datasets under 1000 items

### Performance Tips

- Use local backend for development
- Batch operations when possible
- Consider hybrid approach: develop locally, share via Notion

## 🎯 Best Practices

### 1. Progressive Setup

```python
# Start simple
dataset = Dataset("eval", backend="local/jsonl")

# Migrate when ready
dataset = Dataset("eval", backend="notion")  # Same API!
```

### 2. Error Handling

```python
try:
    dataset = Dataset("my_eval", backend="notion")
    dataset.load()
except FileNotFoundError:
    print("Creating new dataset")
    dataset = Dataset("my_eval", backend="notion")
```

### 3. Data Validation

```python
from pydantic import BaseModel, validator

class EvaluationRecord(BaseModel):
    question: str
    answer: str
    score: float
    
    @validator('score')
    def valid_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be 0-1')
        return v
```

## 🔐 Security

- Keep integration tokens secure (never commit to version control)
- Use environment variables for configuration
- Regularly rotate tokens
- Limit integration permissions to necessary databases

## 🎉 Summary

**Key Benefits:**

- **Local-first development:** Start immediately, migrate when ready
- **Same API:** Code works with any backend
- **Team collaboration:** Rich Notion interface for sharing
- **No vendor lock-in:** Data is portable between backends

**Recommended Flow:**

1. Develop with `backend="local/jsonl"`
2. When ready for team collaboration, set up Notion
3. Switch to `backend="notion"` - same code!
4. Team collaborates via Notion UI while you keep using the API
