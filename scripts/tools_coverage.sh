#!/bin/bash
# tools_coverage.sh - Run comprehensive coverage for all tool modules

set -e

echo "Running comprehensive tools coverage..."

# Clean any existing coverage data
coverage erase

# Test runner tool
echo "=== Running test_runner.py tests ==="
python -m pytest tests/tools/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner

# Task complete tool
echo "=== Running task_complete_tool.py tests ==="
python -m pytest tests/tools/test_task_complete_tool.py -v --cov=src.cli_code.tools.task_complete_tool --cov-append

# File tools
echo "=== Running file_tools.py tests ==="
python -m pytest test_dir/test_file_tools.py -v --cov=src.cli_code.tools.file_tools --cov-append

# Directory tools
echo "=== Running directory_tools.py tests ==="
python -m pytest test_dir/test_directory_tools.py -v --cov=src.cli_code.tools.directory_tools --cov-append

# Quality tools
echo "=== Running quality_tools.py tests ==="
python -m pytest test_dir/improved/test_quality_tools.py -v --cov=src.cli_code.tools.quality_tools --cov-append

# Summarizer tool
echo "=== Running summarizer_tool.py tests ==="
python -m pytest test_dir/improved/test_summarizer_tool.py -v --cov=src.cli_code.tools.summarizer_tool --cov-append

# Tree tool
echo "=== Running tree_tool.py tests ==="
python -m pytest test_dir/improved/test_tree_tool.py test_dir/test_tree_tool_edge_cases.py -v --cov=src.cli_code.tools.tree_tool --cov-append

# System tools
echo "=== Running system_tools.py tests ==="
python -m pytest test_dir/test_tools_basic.py -v --cov=src.cli_code.tools.system_tools --cov-append

# Base tool class
echo "=== Running base.py tests ==="
python -m pytest test_dir/test_tools_init_coverage.py tests/tools/test_base_tool.py -v --cov=src.cli_code.tools.base --cov-append

# Generate comprehensive report
echo "=== Generating comprehensive coverage report ==="
coverage report --include="src/cli_code/tools/*.py"
coverage html

echo "Tools coverage complete. Check coverage_html/index.html for detailed report." 