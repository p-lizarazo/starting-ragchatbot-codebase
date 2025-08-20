#!/usr/bin/env python3
"""
Code formatting script for the RAG chatbot codebase.
Automatically formats code using black and isort.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful, False otherwise."""
    print(f"\n[FORMAT] {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"[DONE] {description} completed")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"[FAIL] {description} failed")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"[FAIL] {description} failed with exception: {e}")
        return False


def main():
    """Format all code files."""
    print("Starting code formatting...")
    
    formatting_commands = [
        ("uv run isort backend/ main.py", "Sorting imports with isort"),
        ("uv run black backend/ main.py", "Formatting code with black"),
    ]
    
    all_passed = True
    
    for command, description in formatting_commands:
        if not run_command(command, description):
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("Code formatting completed successfully!")
        print("\nRun quality checks with: uv run python scripts/quality_check.py")
        sys.exit(0)
    else:
        print("Code formatting failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()