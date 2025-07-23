"""
CLI entry point for ragas webview.
"""

import typer
import sys
import os
from pathlib import Path
from .bundled_server.core.common import DEFAULT_BUNDLED_SERVER_PORT, DEFAULT_BUNDLED_SERVER_HOST


def main(
    directory: str = typer.Argument(..., help="Directory path to serve and visualize"),
    port: int = typer.Option(DEFAULT_BUNDLED_SERVER_PORT, "--port", "-p", help="Port number for the server")
) -> None:
    """
    Start ragas webview server for a directory.

    Args:
        view: Path to the directory containing files to visualize (absolute or relative)
        port: Port number for the server
    """
    try:
        # Handle absolute vs relative path
        if Path(directory).is_absolute():
            directory_path = Path(directory)
        else:
            directory_path = Path.cwd() / directory

        # Resolve to absolute path
        resolved_path = directory_path.resolve()

        # Validate directory exists
        if not resolved_path.exists():
            print(f"❌ Error: Directory does not exist: {resolved_path}")
            sys.exit(1)

        if not resolved_path.is_dir():
            print(f"❌ Error: Path is not a directory: {resolved_path}")
            sys.exit(1)

        # Set environment variables for bundled server
        os.environ["WEBVIEW_PROJECT_DIR"] = str(resolved_path)
        os.environ["FASTAPI_PORT"] = str(port)

        if "ENVIRONMENT" not in os.environ:
            os.environ["ENVIRONMENT"] = "production"

        # Display startup information
        print(f"🚀 Starting Ragas Webview")
        print(f"📁 Project Directory: {resolved_path}")
        print(f"🌐 Server: http://{DEFAULT_BUNDLED_SERVER_HOST}:{port}")

        # Import and start bundled server
        from .bundled_server.main import start_bundled_server
        start_bundled_server()

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("🛑 Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
