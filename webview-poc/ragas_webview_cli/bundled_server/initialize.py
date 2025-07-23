from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .boot import app_config
from .core.common import APP_IDENTIFIER, DEFAULT_JS_BUNDLE_DIR
from .core.exception_handlers import global_exception_handler

# Services  
bundled_server = FastAPI(title=APP_IDENTIFIER, port=app_config.get_port())

# Add CORS middleware to allow cross-origin requests
bundled_server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bundled_server.add_exception_handler(Exception, global_exception_handler)
bundled_server.add_exception_handler(HTTPException, global_exception_handler)

# Store js_bundle_dir for later mounting in main.py after API routes
js_bundle_dir = None
project_dir = app_config.get_project_dir()
if project_dir:
    js_bundle_dir = Path(project_dir).parent / DEFAULT_JS_BUNDLE_DIR

# Export app_config and js_bundle_dir for use in main.py
__all__ = ["bundled_server", "app_config", "js_bundle_dir"]
