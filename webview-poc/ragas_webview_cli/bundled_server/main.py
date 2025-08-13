import uvicorn

from .api.router import api_router
from .core.common import APP_IDENTIFIER, DEFAULT_BUNDLED_SERVER_HOST
from .initialize import app_config, js_bundle_dir
from .initialize import bundled_server as app

# Add API routes first
app.include_router(api_router)

# Mount static files AFTER API routes so API routes take precedence
if js_bundle_dir and js_bundle_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(js_bundle_dir), html=True), name="static")

# Export for uvicorn
bundled_server = app


def start_bundled_server():
    """Start the bundled server with configuration from app_config."""
    uvicorn.run(
        APP_IDENTIFIER,
        host=DEFAULT_BUNDLED_SERVER_HOST,
        port=app_config.get_port(),
        log_level=app_config.get_log_level_value().lower(),
        reload=app_config.reload_uvicorn(),
        use_colors=True,
        timeout_keep_alive=60,  # 60s
        timeout_graceful_shutdown=60,  # 60s final grace period
        limit_concurrency=100,  # Prevent server overload
        backlog=2048,  # Connection queue size
    )


if __name__ == "__main__":
    start_bundled_server()
