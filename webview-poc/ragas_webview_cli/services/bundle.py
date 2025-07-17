"""
Bundle service for handling frontend bundle detection and validation.
Contains both routes and helpers for bundle-related functionality.
"""

from pathlib import Path
from typing import Tuple


def get_js_bundle_path() -> Path:
    """Get the path to the js-bundle directory."""
    # Calculate relative to the services directory
    return Path(__file__).parent.parent.parent / "js-bundle"


def check_bundle_exists() -> Tuple[bool, Path]:
    """Check if the frontend bundle exists and return the path."""
    bundle_path = get_js_bundle_path()
    return bundle_path.exists(), bundle_path


def validate_bundle(bundle_path: Path) -> bool:
    """Validate that the bundle has all required files."""
    if not bundle_path.exists():
        return False
    
    # Check for required files
    index_file = bundle_path / "index.html"
    return index_file.exists()


def print_bundle_status_message() -> None:
    """Print helpful message when bundle is missing."""
    print("❌ Frontend bundle not found!")
    print("📦 The React frontend has not been built yet.")
    print("")
    print("🔨 Please run the build script first:")
    print("   uv run python scripts/build.py")
    print("")
    print("💡 This will create the js-bundle directory with the React app.")
    print("   After building, you can run this CLI again.")
    print("")