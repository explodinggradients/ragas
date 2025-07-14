"""
Configuration constants for Ragas Webview.
"""

import os
from enum import Enum

# Server ports
BACKEND_PORT = 8000
FRONTEND_PORT = 3000

# Server hosts
BACKEND_HOST = "localhost"
FRONTEND_HOST = "localhost"

# URLs
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
FRONTEND_URL = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"

# Environment configuration
class Environment(Enum):
    DEVELOPMENT = "DEVELOPMENT"
    PRODUCTION = "PRODUCTION"

class ReloadMode(Enum):
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"

def get_environment() -> Environment:
    """Get current environment from env var."""
    env_str = os.getenv("WEBVIEW_ENV", "PRODUCTION").upper()
    try:
        return Environment(env_str)
    except ValueError:
        return Environment.PRODUCTION

def is_development() -> bool:
    """Check if running in development mode."""
    return get_environment() == Environment.DEVELOPMENT

def should_reload() -> bool:
    """Check if auto-reload should be enabled."""
    reload_str = os.getenv("WEBVIEW_RELOAD", "DISABLED").upper()
    try:
        reload_mode = ReloadMode(reload_str)
        return reload_mode == ReloadMode.ENABLED
    except ValueError:
        return False