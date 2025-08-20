#!/usr/bin/env python3
"""
Quality check script for the RAG chatbot codebase.
Runs all code quality tools: black, isort, flake8, and mypy.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful, False otherwise."""
    print(f"\n[CHECK] {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"[PASS] {description} passed")
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
    """Run all quality checks."""
    print("Starting code quality checks...")
    
    checks = [
        ("uv run black backend/ main.py --check", "Black formatting check"),
        ("uv run isort backend/ main.py --check-only", "Import sorting check"),
        ("uv run flake8 backend/ main.py", "Flake8 linting"),
        ("uv run mypy backend/ main.py", "MyPy type checking"),
    ]
    
    all_passed = True
    
    for command, description in checks:
        if not run_command(command, description):
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("All quality checks passed!")
        sys.exit(0)
    else:
        print("Some quality checks failed!")
        print("\nTo fix formatting issues, run:")
        print("  uv run black backend/ main.py")
        print("  uv run isort backend/ main.py")
        sys.exit(1)


if __name__ == "__main__":
    main()