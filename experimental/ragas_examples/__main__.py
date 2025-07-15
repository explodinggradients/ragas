"""
CLI interface for running ragas examples.

Usage:
    python -m ragas_examples <example_name>
    python -m ragas_examples --list
    python -m ragas_examples --help
"""

import argparse
import asyncio
import sys
from typing import Dict, Callable, Any


async def run_rag_example():
    """Run RAG evaluation example."""
    from .rag_eval.evals import main
    print("Running RAG evaluation example...")
    await main()


async def run_agent_example():
    """Run agent evaluation example."""
    from .agent_evals.evals import main
    print("Running agent evaluation example...")
    await main()


async def run_prompt_example():
    """Run prompt evaluation example."""
    from .prompt_evals.evals import main
    print("Running prompt evaluation example...")
    await main()


async def run_workflow_example():
    """Run workflow evaluation example."""
    from .workflow_eval.evals import main
    print("Running workflow evaluation example...")
    await main()


EXAMPLES: Dict[str, Callable[[], Any]] = {
    "rag": run_rag_example,
    "agent": run_agent_example,
    "prompt": run_prompt_example,
    "workflow": run_workflow_example,
}


def list_examples():
    """List all available examples."""
    print("Available examples:")
    for name, func in EXAMPLES.items():
        doc = func.__doc__ or "No description available"
        print(f"  {name:10} - {doc.strip()}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Ragas example evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ragas_examples rag        # Run RAG evaluation
    python -m ragas_examples agent      # Run agent evaluation
    python -m ragas_examples prompt     # Run prompt evaluation
    python -m ragas_examples workflow   # Run workflow evaluation
    python -m ragas_examples --list     # List all examples
        """
    )
    
    parser.add_argument(
        "example",
        nargs="?",
        choices=list(EXAMPLES.keys()),
        help="Name of the example to run"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available examples"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_examples()
        return
    
    if not args.example:
        parser.print_help()
        return
    
    if args.example not in EXAMPLES:
        print(f"Unknown example: {args.example}")
        print("Use --list to see available examples")
        sys.exit(1)
    
    try:
        # Run the selected example
        example_func = EXAMPLES[args.example]
        asyncio.run(example_func())
        print(f"\n✅ {args.example} example completed successfully!")
    except KeyboardInterrupt:
        print(f"\n❌ {args.example} example interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ {args.example} example failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()