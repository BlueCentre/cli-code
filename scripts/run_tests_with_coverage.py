#!/usr/bin/env python
"""
Script to run tests with coverage reporting.

This script makes it easy to run the test suite with coverage reporting
and see which parts of the code need more test coverage.

Usage:
    python run_tests_with_coverage.py
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run tests with coverage reporting")
    parser.add_argument("--html", action="store_true", help="Open HTML report after running")
    parser.add_argument("--xml", action="store_true", help="Generate XML report")
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip running tests and just report on existing coverage data"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Get the root directory of the project
    root_dir = Path(__file__).parent

    # Change to the root directory
    os.chdir(root_dir)

    # Add the src directory to Python path to ensure proper imports
    sys.path.insert(0, str(root_dir / "src"))

    if not args.skip_tests:
        # Ensure we have the necessary packages
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], check=False)

        # Run pytest with coverage
        print("\nRunning tests with coverage...")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=cli_code",
            "--cov-report=term",
        ]

        # Add XML report if requested
        if args.xml:
            cmd.append("--cov-report=xml")

        # Always generate HTML report
        cmd.append("--cov-report=html")

        # Add verbosity if requested
        if args.verbose:
            cmd.append("-v")

        # Run tests
        result = subprocess.run(cmd + ["tests/"], check=False)

        if result.returncode != 0:
            print("\n⚠️  Some tests failed! See above for details.")
        else:
            print("\n✅ All tests passed!")

    # Parse coverage results
    try:
        html_report = root_dir / "coverage_html" / "index.html"

        if html_report.exists():
            if args.html:
                print(f"\nOpening HTML coverage report: {html_report}")
                webbrowser.open(f"file://{html_report.absolute()}")
            else:
                print(f"\nHTML coverage report available at: file://{html_report.absolute()}")

        xml_report = root_dir / "coverage.xml"
        if args.xml and xml_report.exists():
            print(f"XML coverage report available at: {xml_report}")

    except Exception as e:
        print(f"Error processing coverage reports: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
