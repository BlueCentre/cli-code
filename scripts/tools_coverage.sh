#!/bin/bash
# Script to run coverage specifically for the tools module

set -e
echo "Running coverage analysis for tools..."

# Clear previous coverage data
coverage erase

# Run tests for each tool, appending coverage data
echo "Testing file_tools..."
python -m pytest tests/tools/test_file_tools.py -v --cov=src.cli_code.tools.file_tools --cov-append

echo "Testing directory_tools..."
python -m pytest tests/tools/test_directory_tools.py -v --cov=src.cli_code.tools.directory_tools --cov-append

echo "Testing quality_tools..."
python -m pytest tests/tools/test_quality_tools.py -v --cov=src.cli_code.tools.quality_tools --cov-append

echo "Testing summarizer_tool..."
python -m pytest tests/tools/test_summarizer_tool.py -v --cov=src.cli_code.tools.summarizer_tool --cov-append

echo "Testing tree_tool..."
python -m pytest tests/tools/test_tree_tool.py tests/tools/test_tree_tool_edge_cases.py -v --cov=src.cli_code.tools.tree_tool --cov-append

echo "Testing system_tools..."
python -m pytest tests/tools/test_system_tools.py -v --cov=src.cli_code.tools.system_tools --cov-append

echo "Testing base_tool and init..."
# Assuming test_tools_init_coverage is in root tests
python -m pytest tests/tools/test_base_tool.py tests/test_tools_init_coverage.py -v --cov=src.cli_code.tools.base --cov=src.cli_code.tools.__init__ --cov-append

echo "Testing task_complete_tool..."
python -m pytest tests/tools/test_task_complete_tool.py -v --cov=src.cli_code.tools.task_complete_tool --cov-append

echo "Testing test_runner_tool..."
python -m pytest tests/tools/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner --cov-append

# Generate final report for the tools module
echo "Generating final report for tools module..."
coverage combine
coverage report --include="src/cli_code/tools/*"
coverage html --include="src/cli_code/tools/*"

echo "Tools coverage complete!"
