#!/usr/bin/env python3
"""
Build script to bundle React app into standalone files.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def check_dependencies():
    """Check if required tools are available."""
    # Check for npm
    if not shutil.which("npm"):
        print("❌ npm not found! Please install Node.js")
        sys.exit(1)
    
    print("✅ Dependencies check passed")


def build_react_app():
    """Build the React app using Vite."""
    react_dir = Path("ragas-webview")
    
    if not react_dir.exists():
        print("❌ React directory 'ragas-webview' not found!")
        sys.exit(1)
    
    print("🔨 Building React app...")
    
    # Check if node_modules exists
    if not (react_dir / "node_modules").exists():
        print("📦 Installing React dependencies...")
        install_result = subprocess.run(
            ["npm", "install"],
            cwd=react_dir,
            capture_output=True,
            text=True
        )
        if install_result.returncode != 0:
            print(f"❌ npm install failed: {install_result.stderr}")
            sys.exit(1)
    
    # Run npm build
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=react_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Build failed: {result.stderr}")
        sys.exit(1)
    
    print("✅ React app built successfully!")
    
    # Check if build output exists
    js_bundle_dir = Path("js-bundle")
    if js_bundle_dir.exists():
        print(f"📦 Build output saved to: {js_bundle_dir.absolute()}")
        
        # List generated files
        print("\n📁 Generated files:")
        for file in js_bundle_dir.rglob("*"):
            if file.is_file():
                print(f"  - {file.relative_to(js_bundle_dir)}")
    else:
        print("❌ Build output directory not found!")
        sys.exit(1)


def main():
    """Main build function."""
    print("🚀 Building Ragas Webview Bundle")
    print("=" * 40)
    
    # Ensure we're in the right directory
    if not Path("ragas-webview").exists():
        print("❌ Please run this from the webview-poc directory")
        sys.exit(1)
    
    # Check dependencies
    check_dependencies()
    
    # Build React app
    build_react_app()
    
    print("\n✅ Build completed successfully!")
    print("💡 Run with uv: uv run ragas-webview-cli")
    print("💡 Run with python: python -m ragas_webview_cli.cli")


if __name__ == "__main__":
    main()