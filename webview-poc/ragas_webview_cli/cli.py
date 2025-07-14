"""
CLI for ragas webview.
"""

import typer
from typing_extensions import Annotated
from .server import start_server
from .config import BACKEND_HOST, BACKEND_PORT


def main(
    port: Annotated[int, typer.Option(help="Port to run server on")] = BACKEND_PORT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = BACKEND_HOST,
    data_dir: Annotated[str, typer.Option(help="Directory containing datasets to serve")] = ".",
):
    """Start ragas webview server."""
    print(f"🚀 Starting Ragas Webview CLI")
    print(f"📍 Host: {host}, Port: {port}")
    print(f"📁 Data directory: {data_dir}")
    start_server(host=host, port=port, data_dir=data_dir)


if __name__ == "__main__":
    typer.run(main)