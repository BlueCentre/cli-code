#!/usr/bin/env python3
"""
Script to analyze coverage data and identify modules with low coverage.
"""

import xml.etree.ElementTree as ET
import sys
import os

# Set the minimum acceptable coverage percentage
MIN_COVERAGE = 60.0

# Check for rich library and provide fallback if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' package for better formatted output: pip install rich")

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

def calculate_module_coverage(root):
    """Calculate coverage percentage for each module."""
    modules = []
    
    # Process packages and classes
    for package in root.findall(".//package"):
        package_name = package.attrib.get("name", "")
        
        for class_elem in package.findall(".//class"):
            filename = class_elem.attrib.get("filename", "")
            line_rate = float(class_elem.attrib.get("line-rate", 0)) * 100
            
            # Count lines covered/valid
            lines = class_elem.find("lines")
            if lines is not None:
                line_count = len(lines.findall("line"))
                covered_count = len([line for line in lines.findall("line") if int(line.attrib.get("hits", 0)) > 0])
            else:
                line_count = 0
                covered_count = 0
            
            modules.append({
                "package": package_name,
                "filename": filename,
                "coverage": line_rate,
                "line_count": line_count,
                "covered_count": covered_count
            })
    
    return modules

def display_coverage_table_rich(modules, min_coverage=MIN_COVERAGE):
    """Display a table of module coverage using rich library."""
    console = Console()
    
    # Create a table
    table = Table(title="Module Coverage Report", box=box.ROUNDED)
    table.add_column("Module", style="cyan")
    table.add_column("Coverage", justify="right", style="green")
    table.add_column("Lines", justify="right", style="blue")
    table.add_column("Covered", justify="right", style="green")
    table.add_column("Missing", justify="right", style="red")
    
    # Sort modules by coverage (ascending)
    modules.sort(key=lambda x: x["coverage"])
    
    # Add modules to table
    for module in modules:
        table.add_row(
            module["filename"],
            f"{module['coverage']:.2f}%",
            str(module["line_count"]),
            str(module["covered_count"]),
            str(module["line_count"] - module["covered_count"]),
            style="red" if module["coverage"] < min_coverage else None
        )
    
    console.print(table)
    
    # Print summary
    below_threshold = [m for m in modules if m["coverage"] < min_coverage]
    console.print(f"\n[bold cyan]Summary:[/]")
    console.print(f"Total modules: [bold]{len(modules)}[/]")
    console.print(f"Modules below {min_coverage}% coverage: [bold red]{len(below_threshold)}[/]")
    
    if below_threshold:
        console.print("\n[bold red]Modules needing improvement:[/]")
        for module in below_threshold:
            console.print(f"  • [red]{module['filename']}[/] ([yellow]{module['coverage']:.2f}%[/])")

def display_coverage_table_plain(modules, min_coverage=MIN_COVERAGE):
    """Display a table of module coverage using plain text."""
    # Calculate column widths
    module_width = max(len(m["filename"]) for m in modules) + 2
    
    # Print header
    print("\nModule Coverage Report")
    print("=" * 80)
    print(f"{'Module':<{module_width}} {'Coverage':>10} {'Lines':>8} {'Covered':>8} {'Missing':>8}")
    print("-" * 80)
    
    # Sort modules by coverage (ascending)
    modules.sort(key=lambda x: x["coverage"])
    
    # Print modules
    for module in modules:
        print(f"{module['filename']:<{module_width}} {module['coverage']:>9.2f}% {module['line_count']:>8} {module['covered_count']:>8} {module['line_count'] - module['covered_count']:>8}")
    
    print("=" * 80)
    
    # Print summary
    below_threshold = [m for m in modules if m["coverage"] < min_coverage]
    print(f"\nSummary:")
    print(f"Total modules: {len(modules)}")
    print(f"Modules below {min_coverage}% coverage: {len(below_threshold)}")
    
    if below_threshold:
        print(f"\nModules needing improvement:")
        for module in below_threshold:
            print(f"  • {module['filename']} ({module['coverage']:.2f}%)")

def main():
    """Main function to analyze coverage data."""
    # Check if coverage.xml exists
    if not os.path.exists("coverage.xml"):
        print("Error: coverage.xml not found. Run coverage tests first.")
        print("Run: ./run_coverage.sh")
        sys.exit(1)
        
    root = parse_coverage_xml()
    overall_coverage = float(root.attrib.get('line-rate', 0)) * 100
    
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"\n[bold cyan]Overall Coverage:[/] [{'green' if overall_coverage >= MIN_COVERAGE else 'red'}]{overall_coverage:.2f}%[/]")
        
        modules = calculate_module_coverage(root)
        display_coverage_table_rich(modules)
    else:
        print(f"\nOverall Coverage: {overall_coverage:.2f}%")
        
        modules = calculate_module_coverage(root)
        display_coverage_table_plain(modules)

if __name__ == "__main__":
    main() 