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
import random
import typing as t
from pathlib import Path

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


class MemorableNames:
    """Generator for memorable, unique names for experiments and datasets."""

    def __init__(self):
        # List of adjectives (similar to what Docker uses)
        self.adjectives = [
            "admiring",
            "adoring",
            "affectionate",
            "agitated",
            "amazing",
            "angry",
            "awesome",
            "blissful",
            "bold",
            "boring",
            "brave",
            "busy",
            "charming",
            "clever",
            "cool",
            "compassionate",
            "competent",
            "condescending",
            "confident",
            "cranky",
            "crazy",
            "dazzling",
            "determined",
            "distracted",
            "dreamy",
            "eager",
            "ecstatic",
            "elastic",
            "elated",
            "elegant",
            "eloquent",
            "epic",
            "fervent",
            "festive",
            "flamboyant",
            "focused",
            "friendly",
            "frosty",
            "gallant",
            "gifted",
            "goofy",
            "gracious",
            "happy",
            "hardcore",
            "heuristic",
            "hopeful",
            "hungry",
            "infallible",
            "inspiring",
            "jolly",
            "jovial",
            "keen",
            "kind",
            "laughing",
            "loving",
            "lucid",
            "magical",
            "mystifying",
            "modest",
            "musing",
            "naughty",
            "nervous",
            "nifty",
            "nostalgic",
            "objective",
            "optimistic",
            "peaceful",
            "pedantic",
            "pensive",
            "practical",
            "priceless",
            "quirky",
            "quizzical",
            "relaxed",
            "reverent",
            "romantic",
            "sad",
            "serene",
            "sharp",
            "silly",
            "sleepy",
            "stoic",
            "stupefied",
            "suspicious",
            "sweet",
            "tender",
            "thirsty",
            "trusting",
            "upbeat",
            "vibrant",
            "vigilant",
            "vigorous",
            "wizardly",
            "wonderful",
            "xenodochial",
            "youthful",
            "zealous",
            "zen",
        ]

        # List of influential computer scientists and tech entrepreneurs
        self.scientists = [
            "turing",
            "hopper",
            "knuth",
            "torvalds",
            "ritchie",
            "thompson",
            "dijkstra",
            "kay",
            "wozniak",
            "gates",
            "jobs",
            "musk",
            "bezos",
            "lovelace",
            "berners_lee",
            "cerf",
            "gosling",
            "kernighan",
            "lamport",
            "mccarthy",
            "minsky",
            "rossum",
            "backus",
            "engelbart",
            "hamilton",
            "chomsky",
            "shannon",
            "zuckerberg",
            "page",
            "brin",
            "matsumoto",
            "stallman",
            "stroustrup",
            "cook",
            "neumann",
            "babbage",
            "tanenbaum",
            "rivest",
            "shamir",
            "adleman",
            "carmack",
            "andreessen",
            "ullman",
            "postel",
            "huffman",
            "boole",
            "curry",
            "liskov",
            "wing",
            "goldwasser",
            "hoare",
            "milner",
            "perlis",
            "sutherland",
            "tarjan",
            "valiant",
            "yao",
            "hopcroft",
            "naur",
            "wilkes",
            "codd",
            "diffie",
            "hellman",
            "pearl",
            "thiel",
            "narayen",
            "nadella",
            "pichai",
            "dorsey",
        ]

        self.used_names = set()

    def generate_name(self):
        """Generate a single memorable name."""
        adjective = random.choice(self.adjectives)
        scientist = random.choice(self.scientists)
        return f"{adjective}_{scientist}"

    def generate_unique_name(self):
        """Generate a unique memorable name."""
        attempts = 0
        max_attempts = 100  # Prevent infinite loops

        while attempts < max_attempts:
            name = self.generate_name()
            if name not in self.used_names:
                self.used_names.add(name)
                return name
            attempts += 1

        # If we exhaust our combinations, add a random suffix
        base_name = self.generate_name()
        unique_name = f"{base_name}_{random.randint(1000, 9999)}"
        self.used_names.add(unique_name)
        return unique_name

    def generate_unique_names(self, count):
        """Generate multiple unique memorable names."""
        return [self.generate_unique_name() for _ in range(count)]


# Global instance for easy access
memorable_names = MemorableNames()


def find_git_root(start_path: t.Union[str, Path, None] = None) -> Path:
    """Find the root directory of a git repository by traversing up from the start path."""
    # Start from the current directory if no path is provided
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()

    # Check if the current directory is a git repository
    current_path = start_path
    while current_path != current_path.parent:  # Stop at filesystem root
        if (current_path / ".git").exists() and (current_path / ".git").is_dir():
            return current_path

        # Move up to the parent directory
        current_path = current_path.parent

    # Final check for the root directory
    if (current_path / ".git").exists() and (current_path / ".git").is_dir():
        return current_path

    # No git repository found
    raise ValueError(f"No git repository found in or above {start_path}")
