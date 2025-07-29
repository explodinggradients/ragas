"""Notion backend implementation for projects and datasets."""

import json
import logging
import os
import typing as t
from datetime import datetime

from pydantic import BaseModel

from .base import BaseBackend

logger = logging.getLogger(__name__)

# Optional dependency handling
try:
    from notion_client import Client as NotionClient
    from notion_client.errors import APIResponseError
    NOTION_AVAILABLE = True
except ImportError:
    NotionClient = None
    APIResponseError = Exception
    NOTION_AVAILABLE = False


class NotionBackend(BaseBackend):
    """Backend for storing datasets and experiments in Notion databases.

    This backend stores data in a single Notion database using properties to organize
    datasets and experiments. Each row represents an item with metadata and data stored
    as JSON in rich text fields.

    Database Structure:
        Required Properties:
        - Name (Title): The name of the dataset/experiment
        - Type (Select): "dataset" or "experiment" 
        - Item_Name (Rich Text): Name/ID of individual items
        - Data (Rich Text): JSON-serialized data for each item
        - Created_At (Date): When the item was created
        - Updated_At (Date): When the item was last updated

    Args:
        token: Notion integration token (or set NOTION_TOKEN env var)
        database_id: ID of the Notion database (or set NOTION_DATABASE_ID env var)

    Environment Variables:
        NOTION_TOKEN: Notion integration token
        NOTION_DATABASE_ID: ID of the target Notion database

    Limitations:
        - Notion API rate limits (3 requests per second)
        - Rich text properties limited to ~2000 characters
        - Large datasets may hit Notion's property limits
        - Complex nested data structures stored as JSON strings

    Best For:
        - Small to medium datasets that fit within Notion's limits
        - Teams already using Notion for documentation/collaboration
        - When you want human-readable data storage with rich formatting
    """

    def __init__(
        self,
        token: t.Optional[str] = None,
        database_id: t.Optional[str] = None,
    ):
        if not NOTION_AVAILABLE:
            raise ImportError(
                "Notion backend requires additional dependencies. "
                "Install with: pip install notion-client"
            )

        # Get from environment if not provided
        self.token = token or os.getenv("NOTION_TOKEN")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")

        if not self.token:
            raise ValueError(
                "Notion token required. Set NOTION_TOKEN environment variable "
                "or pass token parameter."
            )
        
        if not self.database_id:
            raise ValueError(
                "Notion database ID required. Set NOTION_DATABASE_ID environment variable "
                "or pass database_id parameter."
            )

        # Initialize Notion client
        self.client = NotionClient(auth=self.token)
        
        # Validate database access and structure
        self._validate_database()

    @classmethod
    def from_oauth(
        cls, 
        client_id: t.Optional[str] = None,
        client_secret: t.Optional[str] = None,
        redirect_uri: t.Optional[str] = None
    ) -> "NotionBackend":
        """Create NotionBackend using OAuth flow.
        
        This opens a browser for user authorization and automatically
        sets up the database if needed.
        
        Args:
            client_id: OAuth client ID (optional, uses env var if not provided)
            client_secret: OAuth client secret (optional, uses env var if not provided) 
            redirect_uri: OAuth redirect URI (optional, defaults to localhost)
            
        Returns:
            NotionBackend instance ready to use
            
        Raises:
            ImportError: If additional dependencies needed for OAuth
            ValueError: If OAuth flow fails
        """
        try:
            import webbrowser
            import urllib.parse
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            import time
        except ImportError:
            raise ImportError(
                "OAuth flow requires additional dependencies. "
                "For now, use manual setup with integration tokens."
            )
        
        # Get OAuth credentials
        client_id = client_id or os.getenv("NOTION_CLIENT_ID")
        client_secret = client_secret or os.getenv("NOTION_CLIENT_SECRET") 
        redirect_uri = redirect_uri or "http://localhost:8080/callback"
        
        if not client_id:
            # Fallback to simplified setup for now
            logger.warning(
                "OAuth not fully configured. See docs for manual setup with integration tokens."
            )
            raise ValueError(
                "OAuth setup incomplete. Please use manual integration token setup for now.\n"
                "See documentation for NOTION_TOKEN and NOTION_DATABASE_ID setup."
            )
        
        # TODO: Implement full OAuth flow
        # For now, this is a placeholder that guides users to manual setup
        raise NotImplementedError(
            "Full OAuth flow coming soon! For now, please use:\n"
            "1. Create Notion integration at developers.notion.com\n"
            "2. Set NOTION_TOKEN and NOTION_DATABASE_ID environment variables\n"
            "3. Use NotionBackend() constructor"
        )

    def _validate_database(self) -> None:
        """Validate database access and structure."""
        try:
            database = self.client.databases.retrieve(database_id=self.database_id)
        except (APIResponseError, Exception) as e:
            raise ValueError(
                f"Cannot access Notion database {self.database_id}: {e}"
            )

        # Check for required properties
        properties = database.get("properties", {})
        required_props = {
            "Name": "title",
            "Type": "select", 
            "Item_Name": "rich_text",
            "Data": "rich_text",
            "Created_At": "date",
            "Updated_At": "date"
        }

        missing_props = []
        for prop_name, prop_type in required_props.items():
            if prop_name not in properties:
                missing_props.append(f"{prop_name} ({prop_type})")
            elif properties[prop_name]["type"] != prop_type:
                existing_type = properties[prop_name]["type"]
                missing_props.append(
                    f"{prop_name} (expected {prop_type}, found {existing_type})"
                )

        if missing_props:
            raise ValueError(
                f"Database missing required properties: {', '.join(missing_props)}. "
                f"Please add these properties to your Notion database."
            )

        # Ensure Type property has correct select options
        type_property = properties["Type"]
        if type_property["type"] == "select":
            options = {opt["name"] for opt in type_property.get("select", {}).get("options", [])}
            required_options = {"dataset", "experiment"}
            missing_options = required_options - options
            if missing_options:
                logger.warning(
                    f"Type property missing options: {', '.join(missing_options)}. "
                    f"These will be created automatically when first used."
                )

    def _convert_to_notion_properties(self, data_dict: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Convert Python data to Notion property format."""
        notion_props = {}
        
        for key, value in data_dict.items():
            if key in ["Name", "Type", "Item_Name", "Data", "Created_At", "Updated_At"]:
                # Handle special properties
                if key == "Name":
                    notion_props[key] = {"title": [{"text": {"content": str(value)}}]}
                elif key == "Type":
                    notion_props[key] = {"select": {"name": str(value)}}
                elif key in ["Item_Name", "Data"]:
                    content = str(value)
                    # Notion rich text has a limit, truncate if needed
                    if len(content) > 2000:
                        content = content[:1997] + "..."
                        logger.warning(f"Truncated {key} content to fit Notion limits")
                    notion_props[key] = {"rich_text": [{"text": {"content": content}}]}
                elif key in ["Created_At", "Updated_At"]:
                    if isinstance(value, str):
                        notion_props[key] = {"date": {"start": value}}
                    elif isinstance(value, datetime):
                        notion_props[key] = {"date": {"start": value.isoformat()}}
            else:
                # Convert other properties to rich text as JSON
                content = json.dumps(value) if not isinstance(value, str) else value
                if len(content) > 2000:
                    content = content[:1997] + "..."
                    logger.warning(f"Truncated {key} content to fit Notion limits")
                notion_props[key] = {"rich_text": [{"text": {"content": content}}]}
                
        return notion_props

    def _convert_from_notion_properties(self, page: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Convert Notion page properties to Python data."""
        properties = page.get("properties", {})
        result = {}
        
        for prop_name, prop_data in properties.items():
            # Handle both actual Notion API format and test format
            prop_type = prop_data.get("type")
            
            # If no type field, infer from the data structure (for test compatibility)
            if prop_type is None:
                if "title" in prop_data:
                    prop_type = "title"
                elif "select" in prop_data:
                    prop_type = "select"
                elif "rich_text" in prop_data:
                    prop_type = "rich_text"
                elif "date" in prop_data:
                    prop_type = "date"
            
            if prop_type == "title":
                title_content = prop_data.get("title", [])
                result[prop_name] = "".join([t.get("plain_text", "") for t in title_content])
            elif prop_type == "select":
                select_data = prop_data.get("select")
                result[prop_name] = select_data.get("name") if select_data else None
            elif prop_type == "rich_text":
                rich_text_content = prop_data.get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text_content])
                
                # Try to parse as JSON for Data field, otherwise keep as string
                if prop_name == "Data" and text:
                    try:
                        result[prop_name] = json.loads(text)
                    except json.JSONDecodeError:
                        result[prop_name] = text
                else:
                    result[prop_name] = text
            elif prop_type == "date":
                date_data = prop_data.get("date")
                result[prop_name] = date_data.get("start") if date_data else None
            else:
                # Handle other property types as best effort
                result[prop_name] = str(prop_data)
                
        return result

    def _query_database(
        self, 
        data_type: str, 
        name: t.Optional[str] = None
    ) -> t.List[t.Dict[str, t.Any]]:
        """Query the Notion database for specific data type and optionally name."""
        # Convert plural to singular to match what we save (datasets -> dataset)
        type_value = data_type[:-1] if data_type.endswith('s') else data_type
        
        filter_conditions = {
            "property": "Type",
            "select": {
                "equals": type_value
            }
        }
        
        if name:
            filter_conditions = {
                "and": [
                    filter_conditions,
                    {
                        "property": "Name", 
                        "title": {
                            "equals": name
                        }
                    }
                ]
            }

        try:
            response = self.client.databases.query(
                database_id=self.database_id,
                filter=filter_conditions,
                sorts=[{"property": "Created_At", "direction": "ascending"}]
            )
            return response.get("results", [])
        except APIResponseError as e:
            logger.error(f"Failed to query Notion database: {e}")
            raise

    def _load(self, data_type: str, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load data from Notion database."""
        pages = self._query_database(data_type, name)
        
        if not pages:
            raise FileNotFoundError(
                f"No {data_type[:-1]} named '{name}' found in Notion database"
            )

        # Convert pages to data format
        result = []
        for page in pages:
            converted = self._convert_from_notion_properties(page)
            if "Data" in converted and converted["Data"]:
                # If Data field contains the actual data, use it
                if isinstance(converted["Data"], dict):
                    result.append(converted["Data"])
                else:
                    # Fallback to using all converted properties
                    result.append(converted)
            else:
                result.append(converted)

        return result

    def _save(
        self,
        data_type: str,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        """Save data to Notion database."""
        # First, delete existing entries for this dataset/experiment
        existing_pages = self._query_database(data_type, name)
        for page in existing_pages:
            try:
                self.client.pages.update(
                    page_id=page["id"],
                    archived=True
                )
            except APIResponseError as e:
                logger.warning(f"Failed to archive existing page: {e}")

        # Create new entries
        current_time = datetime.now().isoformat()
        
        for i, item in enumerate(data):
            # Create properties for the new page
            properties = {
                "Name": name,
                "Type": data_type[:-1],  # Remove 's' from 'datasets'/'experiments'
                "Item_Name": f"{name}_item_{i}",
                "Data": json.dumps(item),
                "Created_At": current_time,
                "Updated_At": current_time
            }
            
            notion_properties = self._convert_to_notion_properties(properties)
            
            try:
                self.client.pages.create(
                    parent={"database_id": self.database_id},
                    properties=notion_properties
                )
            except APIResponseError as e:
                logger.error(f"Failed to create page for item {i}: {e}")
                raise

    def _list(self, data_type: str) -> t.List[str]:
        """List all available datasets or experiments."""
        pages = self._query_database(data_type)
        
        # Extract unique names
        names = set()
        for page in pages:
            converted = self._convert_from_notion_properties(page)
            if "Name" in converted and converted["Name"]:
                names.add(converted["Name"])
        
        return sorted(list(names))

    # Public interface methods (required by BaseBackend)
    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load dataset from Notion database."""
        return self._load("datasets", name)

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load experiment from Notion database."""
        return self._load("experiments", name)

    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save dataset to Notion database."""
        self._save("datasets", name, data, data_model)

    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save experiment to Notion database."""
        self._save("experiments", name, data, data_model)

    def list_datasets(self) -> t.List[str]:
        """List all dataset names."""
        return self._list("datasets")

    def list_experiments(self) -> t.List[str]:
        """List all experiment names."""
        return self._list("experiments")

    def __repr__(self) -> str:
        return f"NotionBackend(database_id='{self.database_id[:8]}...')"

    __str__ = __repr__
