"""Shared utilities for project module."""

import random
import string
import uuid


def create_nano_id(size=12):
    """Create a short, URL-safe unique identifier."""
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
