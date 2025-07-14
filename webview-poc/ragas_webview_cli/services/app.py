"""
SPA (Single Page Application) serving service.
Handles serving React frontend and static files.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


def create_app_router(js_bundle_dir: Path) -> APIRouter:
    """Create router for SPA serving."""
    router = APIRouter()
    
    # Serve index.html for root
    @router.get("/")
    async def serve_app():
        """Serve the main React app."""
        index_file = js_bundle_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Frontend not built")
        return FileResponse(index_file)
    
    # Serve static files only
    @router.get("/{full_path:path}")
    async def serve_static_files(full_path: str):
        """Serve static files from js-bundle."""
        if full_path.startswith("api/"):
            # Let FastAPI handle API routes
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Check if it's a static file in js-bundle root
        static_file = js_bundle_dir / full_path
        if static_file.exists() and static_file.is_file():
            return FileResponse(static_file)
        
        # For non-existent routes, return 404
        raise HTTPException(status_code=404, detail="Not found")
    
    return router


def mount_static_files(app, js_bundle_dir: Path):
    """Mount static files for assets."""
    assets_dir = js_bundle_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")