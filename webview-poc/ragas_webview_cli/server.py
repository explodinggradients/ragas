"""
Main server for Ragas Webview CLI.
Orchestrates different services and creates the FastAPI application.
"""

import os
import uvicorn
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import BACKEND_HOST, BACKEND_PORT, FRONTEND_URL, should_reload, is_development
from .services.app import create_app_router, mount_static_files
from .services.datasets import create_datasets_router
from .services.bundle import get_js_bundle_path, check_bundle_exists, print_bundle_status_message


def create_app(data_dir: Optional[str] = None) -> FastAPI:
    """Create FastAPI application with all services."""
    app = FastAPI(
        title="Ragas Webview",
        description="Web-based file viewer with React frontend and Python backend",
        version="1.0.0"
    )
    
    # Add CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_URL],  # React dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/api/health")
    async def health():
        return {"status": "healthy", "service": "ragas-webview"}
    
    # Include service routes
    datasets_router = create_datasets_router(data_dir)
    app.include_router(datasets_router)
    
    # Static files and SPA serving (production mode)
    js_bundle_dir = get_js_bundle_path()
    bundle_exists, _ = check_bundle_exists()
    
    if bundle_exists:
        print(f"📦 Serving static files from: {js_bundle_dir}")
        
        # Mount static assets
        mount_static_files(app, js_bundle_dir)
        
        # Include SPA router (must be last due to catch-all routes)
        app_router = create_app_router(js_bundle_dir)
        app.include_router(app_router)
    
    else:
        # Development fallback when no build exists
        @app.get("/")
        async def root():
            return {
                "message": "Ragas Webview API", 
                "mode": "development",
                "note": "Run 'uv run python scripts/build.py' to create production bundle",
            }
    
    return app


# Module-level app instance for reload mode
app = create_app(data_dir=os.getenv("WEBVIEW_DATA_DIR"))


def start_server(host: str = BACKEND_HOST, port: int = BACKEND_PORT, data_dir: Optional[str] = None):
    """Start the webview server."""
    
    # Check bundle status and print helpful message if needed
    bundle_exists, _ = check_bundle_exists()
    if not bundle_exists:
        print_bundle_status_message()
    
    app = create_app(data_dir=data_dir)
    
    reload_enabled = should_reload()
    dev_mode = is_development()
    
    print(f"🚀 Starting server at http://{host}:{port}")
    print(f"🔧 Mode: {'Development' if dev_mode else 'Production'}")
    print(f"🔄 Auto-reload: {'Enabled' if reload_enabled else 'Disabled'}")
    if data_dir:
        print(f"📁 Data directory: {Path(data_dir).resolve()}")
    
    if reload_enabled:
        # For reload mode, set data_dir as environment variable
        if data_dir:
            os.environ["WEBVIEW_DATA_DIR"] = data_dir
        
        # Use import string for reload mode
        uvicorn.run(
            "ragas_webview_cli.server:app",
            host=host, 
            port=port,
            reload=True,
            log_level="debug" if dev_mode else "info"
        )
    else:
        # Use app instance for production mode
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            log_level="debug" if dev_mode else "info"
        )