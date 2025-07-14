#!/usr/bin/env python3
"""
Development script to run both Python backend and React frontend servers.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path
from ragas_webview_cli.config import BACKEND_HOST, BACKEND_PORT, FRONTEND_HOST, FRONTEND_PORT, BACKEND_URL, FRONTEND_URL


def run_python_server():
    """Run the Python FastAPI server with development settings."""
    print(f"🐍 Starting Python server on {BACKEND_URL}")
    print("🔄 Auto-reload enabled for development")
    
    # Set environment variables for development mode
    env = os.environ.copy()
    env["WEBVIEW_ENV"] = "DEVELOPMENT"
    env["WEBVIEW_RELOAD"] = "ENABLED"
    
    # Use the installed CLI command to avoid module import conflicts
    return subprocess.Popen([
        "uv", "run", "ragas-webview-cli", "--port", str(BACKEND_PORT), "--host", BACKEND_HOST
    ], env=env)


def run_react_server():
    """Run the React dev server."""
    print(f"⚛️  Starting React dev server on {FRONTEND_URL}")
    react_dir = Path("ragas-webview").resolve()
    
    if not react_dir.exists():
        print("❌ React directory 'ragas-webview' not found!")
        return None
    
    # Set environment variables for React dev server
    env = os.environ.copy()
    env["VITE_API_HOST"] = BACKEND_HOST
    env["VITE_API_PORT"] = str(BACKEND_PORT)
    env["PORT"] = str(FRONTEND_PORT)
    env["HOST"] = FRONTEND_HOST
    
    return subprocess.Popen([
        "npm", "run", "dev"
    ], cwd=react_dir, env=env)


def main():
    """Main function to start both servers."""
    print("🚀 Starting Ragas Webview Development Environment")
    print("=" * 50)
    
    # Start Python server
    python_process = run_python_server()
    time.sleep(2)  # Give Python server time to start
    
    # Start React server
    react_process = run_react_server()
    
    if not react_process:
        python_process.terminate()
        sys.exit(1)
    
    print("\n✅ Both servers started!")
    print(f"📱 Frontend: {FRONTEND_URL}")
    print(f"🔧 Backend API: {BACKEND_URL}")
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