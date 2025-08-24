#!/usr/bin/env python
"""
Script to process experimental notebooks with nbdev and convert to markdown for MkDocs documentation.
This script should be executed from the project root directory.
"""

import os
import subprocess
import sys
from pathlib import Path

# Path configurations
RAGAS_ROOT = Path(__file__).parent.parent
EXPERIMENTAL_DIR = RAGAS_ROOT / "experimental"
PROC_DIR = EXPERIMENTAL_DIR / "_proc"
DOCS_DIR = RAGAS_ROOT / "docs" / "experimental"


def run_command(cmd, cwd=None):
    """Run a shell command and print output"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result.stdout


def process_notebooks():
    """Process notebooks with nbdev_proc_nbs"""
    # Check if there are any notebooks to process
    notebook_files = list(EXPERIMENTAL_DIR.glob("**/*.ipynb"))
    if not notebook_files:
        print("No notebooks found in experimental directory, skipping nbdev processing...")
        return
    
    print("Processing notebooks with nbdev...")
    try:
        run_command(["nbdev_proc_nbs"], cwd=EXPERIMENTAL_DIR)
    except FileNotFoundError:
        print("Warning: nbdev_proc_nbs not found. Install nbdev to process notebooks.")
        return

    if not PROC_DIR.exists():
        print(
            f"Error: Expected processed notebooks at {PROC_DIR}, but directory does not exist."
        )
        sys.exit(1)

    print(f"Notebooks processed successfully to {PROC_DIR}")


def render_with_quarto():
    """Render processed notebooks to markdown using Quarto"""
    # Check if there are notebooks to render
    if not PROC_DIR.exists():
        print("No processed notebooks directory found, skipping Quarto rendering...")
        return
    
    notebook_files = list(PROC_DIR.glob("**/*.ipynb"))
    if not notebook_files:
        print("No notebooks found for rendering, skipping Quarto rendering...")
        return
    
    print("Rendering notebooks to markdown with Quarto...")

    # Ensure the output directory exists
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Run Quarto to convert notebooks to markdown
    try:
        run_command(
            [
                "quarto",
                "render",
                "**/*.ipynb",
                "--to",
                "gfm",
                "--output-dir",
                str(DOCS_DIR),
            ],
            cwd=PROC_DIR,
        )
    except FileNotFoundError:
        print("Warning: quarto not found. Install Quarto CLI to render notebooks.")
        return
    except Exception as e:
        print(f"Error rendering notebooks with Quarto: {e}")
        sys.exit(1)

    print(f"Notebooks rendered successfully to {DOCS_DIR}")


def main():
    """Main function to process notebooks and render to markdown"""
    # Ensure we're in the project root
    if (
        not (RAGAS_ROOT / "src" / "ragas").exists()
        or not (RAGAS_ROOT / "experimental").exists()
    ):
        print("Error: This script must be run from the ragas project root directory.")
        sys.exit(1)

    # Process notebooks with nbdev
    process_notebooks()

    # Render notebooks to markdown with Quarto
    render_with_quarto()

    print("Notebook processing and rendering completed successfully!")


if __name__ == "__main__":
    main()
