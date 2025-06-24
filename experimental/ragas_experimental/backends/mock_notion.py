"""Helps with testing `ragas_annotator` better."""

__all__ = [
    "MockPagesAPI",
    "MockDatabasesAPI",
    "MockBlocksAPI",
    "MockBlockChildrenAPI",
    "MockNotionClient",
]

import uuid
from copy import deepcopy
from datetime import datetime

from ..exceptions import NotFoundError


class MockPagesAPI:
    """Mock implementation of notion_client.Client.pages"""

    def __init__(self, client):
        self.client = client

    def create(self, parent, properties, **kwargs):
        """Create a new page."""
        page_id = self.client._create_id()

        # Create the page object
        page = {
            "id": page_id,
            "object": "page",
            "created_time": self.client._get_timestamp(),
            "last_edited_time": self.client._get_timestamp(),
            "archived": False,
            "properties": deepcopy(properties),
            "parent": deepcopy(parent),
        }

        # Add page to storage
        self.client._pages[page_id] = page

        # Add child reference to parent
        parent_type = parent.get("type")
        parent_id = parent.get(f"{parent_type}_id")

        if parent_id:
            child_block = {
                "id": self.client._create_id(),
                "object": "block",
                "type": "child_page",
                "created_time": self.client._get_timestamp(),
                "last_edited_time": self.client._get_timestamp(),
                "child_page": {"title": self._extract_title(properties)},
            }

            if parent_id not in self.client._children:
                self.client._children[parent_id] = []

            self.client._children[parent_id].append(child_block)

        return deepcopy(page)

    def retrieve(self, page_id):
        """Retrieve a page by ID."""
        if page_id not in self.client._pages:
            raise NotFoundError(f"Page {page_id} not found")

        return deepcopy(self.client._pages[page_id])

    def update(self, page_id, properties=None, archived=None, **kwargs):
        """Update a page."""
        if page_id not in self.client._pages:
            raise NotFoundError(f"Page {page_id} not found")

        page = self.client._pages[page_id]

        if properties:
            # Update properties
            for key, value in properties.items():
                page["properties"][key] = deepcopy(value)

        if archived is not None:
            page["archived"] = archived

        page["last_edited_time"] = self.client._get_timestamp()

        return deepcopy(page)

    def _extract_title(self, properties):
        """Extract page title from properties."""
        for prop in properties.values():
            if prop.get("type") == "title" and prop.get("title"):
                for text_obj in prop["title"]:
                    if text_obj.get("type") == "text" and "content" in text_obj.get(
                        "text", {}
                    ):
                        return text_obj["text"]["content"]
        return "Untitled"


class MockDatabasesAPI:
    """Mock implementation of notion_client.Client.databases"""

    def __init__(self, client):
        self.client = client

    def create(self, parent, title, properties, **kwargs):
        """Create a new database."""
        database_id = self.client._create_id()

        # Create database object
        database = {
            "id": database_id,
            "object": "database",
            "created_time": self.client._get_timestamp(),
            "last_edited_time": self.client._get_timestamp(),
            "title": deepcopy(title),
            "properties": deepcopy(properties),
            "parent": deepcopy(parent),
        }

        # Add database to storage
        self.client._databases[database_id] = database

        # Add child reference to parent
        parent_type = parent.get("type")
        parent_id = parent.get(f"{parent_type}_id")

        if parent_id:
            child_block = {
                "id": self.client._create_id(),
                "object": "block",
                "type": "child_database",
                "created_time": self.client._get_timestamp(),
                "last_edited_time": self.client._get_timestamp(),
                "child_database": {"title": self._extract_title(title)},
            }

            if parent_id not in self.client._children:
                self.client._children[parent_id] = []

            self.client._children[parent_id].append(child_block)

        return deepcopy(database)

    def retrieve(self, database_id):
        """Retrieve a database by ID."""
        if database_id not in self.client._databases:
            raise NotFoundError(f"Database {database_id} not found")

        return deepcopy(self.client._databases[database_id])

    def query(
        self,
        database_id,
        filter=None,
        sorts=None,
        start_cursor=None,
        page_size=100,
        **kwargs,
    ):
        """Query a database."""
        if database_id not in self.client._databases:
            raise NotFoundError(f"Database {database_id} not found")

        # Get all pages in the database
        results = []
        for page_id, page in self.client._pages.items():
            parent = page.get("parent", {})
            if (
                parent.get("type") == "database_id"
                and parent.get("database_id") == database_id
            ):
                results.append(deepcopy(page))

        # TODO: Implement filtering, sorting, and pagination if needed

        return {"results": results, "has_more": False, "next_cursor": None}

    def _extract_title(self, title):
        """Extract database title from title array."""
        for text_obj in title:
            if text_obj.get("type") == "text" and "content" in text_obj.get("text", {}):
                return text_obj["text"]["content"]
        return "Untitled"


class MockBlocksAPI:
    """Mock implementation of notion_client.Client.blocks"""

    def __init__(self, client):
        self.client = client
        self.children = MockBlockChildrenAPI(client)

    def retrieve(self, block_id):
        """Retrieve a block by ID."""
        if block_id not in self.client._blocks:
            raise NotFoundError(f"Block {block_id} not found")

        return deepcopy(self.client._blocks[block_id])


class MockBlockChildrenAPI:
    """Mock implementation of notion_client.Client.blocks.children"""

    def __init__(self, client):
        self.client = client

    def list(self, block_id, start_cursor=None, page_size=100):
        """List children of a block."""
        children = self.client._children.get(block_id, [])

        # TODO: Implement pagination if needed

        return {"results": deepcopy(children), "has_more": False, "next_cursor": None}


class MockNotionClient:
    """Mock implementation of notion_client.Client for testing."""

    def __init__(self, auth=None):
        """Initialize the mock client with in-memory storage.

        Args:
            auth: Ignored in mock implementation
        """
        # In-memory storage
        self._pages = {}  # page_id -> page object
        self._databases = {}  # database_id -> database object
        self._blocks = {}  # block_id -> block object
        self._children = {}  # parent_id -> list of child blocks

        # Create API namespaces to match real client
        self.pages = MockPagesAPI(self)
        self.databases = MockDatabasesAPI(self)
        self.blocks = MockBlocksAPI(self)

    def _get_timestamp(self):
        """Generate a timestamp in Notion API format."""
        return datetime.utcnow().isoformat() + "Z"

    def _create_id(self):
        """Generate a random ID in Notion format."""
        return str(uuid.uuid4()).replace("-", "")

    def add_page(self, page_data):
        """Add a page to the mock storage."""
        self._pages[page_data["id"]] = deepcopy(page_data)

    def add_database(self, database_data):
        """Add a database to the mock storage."""
        self._databases[database_data["id"]] = deepcopy(database_data)

    def add_block(self, block_data):
        """Add a block to the mock storage."""
        self._blocks[block_data["id"]] = deepcopy(block_data)

    def add_children(self, parent_id, children):
        """Add children to a parent."""
        if parent_id not in self._children:
            self._children[parent_id] = []
        self._children[parent_id].extend(deepcopy(children))

    def __str__(self):
        return "MockNotionClient(num_pages={}, num_databases={}, num_blocks={})".format(
            len(self._pages), len(self._databases), len(self._blocks)
        )

    __repr__ = __str__
