#!/usr/bin/env python3
"""
Build script to bundle React app into standalone files.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

# Add the parent directory to Python path so we can import ragas_webview_cli
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas_webview_cli.bundled_server.core.common import DEFAULT_BUNDLED_SERVER_PORT, DEFAULT_JS_BUNDLE_DIR


def check_dependencies():
    """Check if required tools are available."""
    # Check for pnpm
    if not shutil.which("pnpm"):
        print("❌ pnpm not found! Please install pnpm")
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
            ["pnpm", "install"],
            cwd=react_dir,
            capture_output=True,
            text=True
        )
        if install_result.returncode != 0:
            print(f"❌ pnpm install failed: {install_result.stderr}")
            sys.exit(1)
    
    # Run pnpm build
    result = subprocess.run(
        ["pnpm", "run", "build"],
        cwd=react_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Build failed: {result.stderr}")
        sys.exit(1)
    
    print("✅ React app built successfully!")
    
    # Check if build output exists in js-bundle directory
    js_bundle_dir = Path(DEFAULT_JS_BUNDLE_DIR)
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
    print("💡 Run with: python -m ragas_webview_cli <directory> --port <port>")
    print(f"💡 Example: python -m ragas_webview_cli ./logs --port {DEFAULT_BUNDLED_SERVER_PORT}")


if __name__ == "__main__":
    main()