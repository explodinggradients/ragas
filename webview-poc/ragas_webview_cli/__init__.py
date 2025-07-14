"""
Ragas Webview CLI - A web-based file viewer with React frontend and Python backend.
"""

__version__ = "0.1.0"

from .server import create_app, start_server
from .cli import main

__all__ = ["create_app", "start_server", "main"]