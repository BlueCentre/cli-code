#!/bin/bash
# run_targeted_coverage.sh - Run coverage tests for specific modules

# Set error handling
set -e

echo "Running coverage for key modules..."

# Clear existing coverage data first to avoid mixed results
coverage erase

# Test runner tool - Run test files separately to avoid import conflicts
echo "==== Testing test_runner.py (test_dir tests) ===="
python -m pytest test_dir/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner --cov-report=term

echo "==== Testing test_runner.py (tests/tools tests) ===="
python -m pytest tests/tools/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner --cov-append --cov-report=term

# Add test_runner coverage to main report
coverage combine

# Gemini model
echo "==== Testing gemini.py ===="
python -m pytest test_dir/test_gemini_model*.py -v --cov=src.cli_code.models.gemini --cov-report=term

# Ollama model
echo "==== Testing ollama.py ===="
python -m pytest test_dir/test_ollama_model*.py -v --cov=src.cli_code.models.ollama --cov-report=term

# File tools
echo "==== Testing file_tools.py ===="
python -m pytest test_dir/test_file_tools.py -v --cov=src.cli_code.tools.file_tools --cov-report=term

# Tree tool
echo "==== Testing tree_tool.py ===="
python -m pytest test_dir/test_tree_tool.py -v --cov=src.cli_code.tools.tree_tool --cov-report=term
python -m pytest test_dir/test_tree_tool_edge_cases.py -v --cov=src.cli_code.tools.tree_tool --cov-append --cov-report=term

# Task complete tool
echo "==== Testing task_complete_tool.py ===="
python -m pytest test_dir/test_task_complete_tool.py -v --cov=src.cli_code.tools.task_complete_tool --cov-report=term
python -m pytest tests/tools/test_task_complete_tool.py -v --cov=src.cli_code.tools.task_complete_tool --cov-append --cov-report=term

# Run tests for all remaining tools to get comprehensive coverage
echo "==== Testing other tools ===="
python -m pytest test_dir/test_tools_basic.py test_dir/test_tools_init_coverage.py test_dir/test_directory_tools.py test_dir/test_quality_tools.py test_dir/test_summarizer_tool.py -v --cov=src.cli_code.tools --cov-report=term

# Generate a complete coverage report at the end
coverage combine
coverage report
coverage html

echo "Targeted coverage complete!" 