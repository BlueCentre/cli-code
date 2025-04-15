#!/usr/bin/env python3
"""
Script to generate a coverage badge for the README.
This creates a shields.io URL that displays the current coverage.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote

# Default colors for different coverage levels
COLORS = {
    "excellent": "brightgreen",
    "good": "green",
    "acceptable": "yellowgreen",
    "warning": "yellow",
    "poor": "orange",
    "critical": "red",
}


def parse_coverage_xml(file_path="coverage.xml"):
    """Parse the coverage XML file and extract coverage data."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root
    except FileNotFoundError:
        print(f"Error: Coverage file '{file_path}' not found. Run coverage first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing coverage XML: {e}")
        sys.exit(1)


def get_coverage_color(coverage_percent):
    """Determine the appropriate color based on coverage percentage."""
    if coverage_percent >= 90:
        return COLORS["excellent"]
    elif coverage_percent >= 80:
        return COLORS["good"]
    elif coverage_percent >= 70:
        return COLORS["acceptable"]
    elif coverage_percent >= 60:
        return COLORS["warning"]
    elif coverage_percent >= 50:
        return COLORS["poor"]
    else:
        return COLORS["critical"]


def generate_badge_url(coverage_percent, label="coverage", color=None):
    """Generate a shields.io URL for the coverage badge."""
    if color is None:
        color = get_coverage_color(coverage_percent)

    # Format the coverage percentage with 2 decimal places
    coverage_formatted = f"{coverage_percent:.2f}%"

    # Construct the shields.io URL
    url = f"https://img.shields.io/badge/{quote(label)}-{quote(coverage_formatted)}-{color}"
    return url


def generate_badge_markdown(coverage_percent, label="coverage"):
    """Generate markdown for a coverage badge."""
    url = generate_badge_url(coverage_percent, label)
    return f"![{label}]({url})"


def generate_badge_html(coverage_percent, label="coverage"):
    """Generate HTML for a coverage badge."""
    url = generate_badge_url(coverage_percent, label)
    return f'<img src="{url}" alt="{label}">'


def main():
    """Main function to generate coverage badge."""
    parser = argparse.ArgumentParser(description="Generate a coverage badge")
    parser.add_argument(
        "--format", choices=["url", "markdown", "html"], default="markdown", help="Output format (default: markdown)"
    )
    parser.add_argument("--label", default="coverage", help='Badge label (default: "coverage")')
    parser.add_argument("--file", default="coverage.xml", help="Coverage XML file path (default: coverage.xml)")
    args = parser.parse_args()

    # Check if coverage.xml exists
    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found. Run coverage tests first.")
        print("Run: ./run_coverage.sh")
        sys.exit(1)

    # Get coverage percentage
    root = parse_coverage_xml(args.file)
    coverage_percent = float(root.attrib.get("line-rate", 0)) * 100

    # Generate badge in requested format
    if args.format == "url":
        output = generate_badge_url(coverage_percent, args.label)
    elif args.format == "html":
        output = generate_badge_html(coverage_percent, args.label)
    else:  # markdown
        output = generate_badge_markdown(coverage_percent, args.label)

    print(output)


if __name__ == "__main__":
    main()
