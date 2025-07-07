__all__ = [
    "create_nano_id",
    "async_to_sync",
    "get_test_directory",
]

import asyncio
import functools
import os
import string
import tempfile
import uuid

from rich.console import Console

console = Console()


def create_nano_id(size=12):
    # Define characters to use (alphanumeric)
    alphabet = string.ascii_letters + string.digits

    # Generate UUID and convert to int
    uuid_int = uuid.uuid4().int

    # Convert to base62
    result = ""
    while uuid_int:
        uuid_int, remainder = divmod(uuid_int, len(alphabet))
        result = alphabet[remainder] + result

    # Pad if necessary and return desired length
    return result[:size]


def async_to_sync(async_func):
    """Convert an async function to a sync function"""

    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(async_func(*args, **kwargs))

    return sync_wrapper


# Helper function for tests
def get_test_directory():
    """Create a test directory that will be cleaned up on process exit.

    Returns:
        str: Path to test directory
    """
    # Create a directory in the system temp directory
    test_dir = os.path.join(tempfile.gettempdir(), f"ragas_test_{create_nano_id()}")
    os.makedirs(test_dir, exist_ok=True)

    return test_dir
