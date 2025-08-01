#!/usr/bin/env python3
"""
Development script to run both Python backend and React frontend servers.
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

# Add the parent directory to Python path so we can import ragas_webview_cli
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas_webview_cli.bundled_server.core.common import (
    DEFAULT_BUNDLED_SERVER_HOST,
    DEFAULT_BUNDLED_SERVER_PORT,
    DEFAULT_FRONTEND_HOST,
    DEFAULT_FRONTEND_PORT
)


def run_python_server(directory: str):
    """Run the Python FastAPI server with development settings."""
    print(f"🐍 Starting Python server on http://{DEFAULT_BUNDLED_SERVER_HOST}:{DEFAULT_BUNDLED_SERVER_PORT}")
    print(f"📁 Serving directory: {Path(directory).resolve()}")
    print("🔄 Auto-reload enabled for development")
    
    # Set environment variables for development mode
    env = os.environ.copy()
    env["ENVIRONMENT"] = "DEVELOPMENT"
    env["WEBVIEW_PROJECT_DIR"] = str(Path(directory).resolve())
    env["FASTAPI_PORT"] = str(DEFAULT_BUNDLED_SERVER_PORT)
    
    # Use the CLI module with directory argument and port
    return subprocess.Popen([
        "uv", "run", "python", "-m", "ragas_webview_cli", directory, "--port", str(DEFAULT_BUNDLED_SERVER_PORT)
    ], env=env)

def run_react_server():
    """Run the React dev server."""
    print(f"⚛️  Starting React dev server on http://{DEFAULT_FRONTEND_HOST}:{DEFAULT_FRONTEND_PORT}")
    react_dir = Path("ragas-webview").resolve()
    
    if not react_dir.exists():
        print("❌ React directory 'ragas-webview' not found!")
        return None
    
    # Set environment variables for React dev server
    env = os.environ.copy()
    env["VITE_DEFAULT_BUNDLED_SERVER_HOST"] = DEFAULT_BUNDLED_SERVER_HOST
    env["VITE_DEFAULT_BUNDLED_SERVER_PORT"] = str(DEFAULT_BUNDLED_SERVER_PORT)
    env["PORT"] = str(DEFAULT_FRONTEND_PORT)
    env["HOST"] = DEFAULT_FRONTEND_HOST
    
    return subprocess.Popen([
        "pnpm", "run", "dev"
    ], cwd=react_dir, env=env)


def main():
    """Main function to start both servers."""
    parser = argparse.ArgumentParser(description="Start development environment")
    parser.add_argument("directory", help="Directory to serve and visualize")
    args = parser.parse_args()
    
    # Validate directory exists
    directory_path = Path(args.directory)
    if not directory_path.exists():
        print(f"❌ Directory does not exist: {directory_path}")
        sys.exit(1)
    if not directory_path.is_dir():
        print(f"❌ Path is not a directory: {directory_path}")
        sys.exit(1)
    
    print("🚀 Starting Ragas Webview Development Environment")
    print("=" * 50)
    
    # Start Python server
    python_process = run_python_server(args.directory)
    time.sleep(2)  # Give Python server time to start
    
    # Start React server
    react_process = run_react_server()
    
    if not react_process:
        python_process.terminate()
        sys.exit(1)
    
    print("\n✅ Both servers started!")
    print(f"📱 Frontend: http://{DEFAULT_FRONTEND_HOST}:{DEFAULT_FRONTEND_PORT}")
    print(f"🔧 Backend API: http://{DEFAULT_BUNDLED_SERVER_HOST}:{DEFAULT_BUNDLED_SERVER_PORT}")
    print(f"📁 Serving: {directory_path.resolve()}")
    print("\nPress Ctrl+C to stop both servers")
    
    try:
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if processes are still running
            if python_process.poll() is not None:
                print("❌ Python server stopped")
                break
            if react_process.poll() is not None:
                print("❌ React server stopped")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        
        # Terminate processes
        python_process.terminate()
        react_process.terminate()
        
        # Wait for clean shutdown
        python_process.wait()
        react_process.wait()
        
        print("✅ Both servers stopped")


if __name__ == "__main__":
    main()