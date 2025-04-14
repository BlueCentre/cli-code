#!/bin/bash
# Script to run targeted coverage reports for specific modules

set -e
echo "Running targeted coverage analysis..."

# Clear previous coverage data
coverage erase

# --- Test Runner --- 
echo "==== Testing test_runner.py (tests/tools tests) ===="
python -m pytest tests/tools/test_test_runner_tool.py -v --cov=src.cli_code.tools.test_runner --cov-report=term

# Add test_runner coverage to main report
coverage combine

# --- Models --- 
echo "==== Testing Gemini models (tests/models tests) ===="
python -m pytest tests/models/test_gemini*.py -v --cov=src.cli_code.models.gemini --cov-report=term

echo "==== Testing Ollama models (tests/models tests) ===="
python -m pytest tests/models/test_ollama*.py -v --cov=src.cli_code.models.ollama --cov-report=term

# --- Specific Tools --- 
echo "==== Testing File Tools (tests/tools tests) ===="
python -m pytest tests/tools/test_file_tools.py -v --cov=src.cli_code.tools.file_tools --cov-report=term

echo "==== Testing Tree Tool (tests/tools tests) ===="
python -m pytest tests/tools/test_tree_tool.py -v --cov=src.cli_code.tools.tree_tool --cov-report=term
python -m pytest tests/tools/test_tree_tool_edge_cases.py -v --cov=src.cli_code.tools.tree_tool --cov-append --cov-report=term

echo "==== Testing Task Complete Tool (tests/tools tests) ===="
python -m pytest tests/tools/test_task_complete_tool.py -v --cov=src.cli_code.tools.task_complete_tool --cov-report=term

# --- Other Tools --- 
echo "==== Testing Other Tools (tests/tools tests) ===="
# Note: test_tools_init_coverage might be in root tests folder
python -m pytest tests/tools/test_tools_basic.py tests/tools/test_directory_tools.py tests/tools/test_quality_tools.py tests/tools/test_summarizer_tool.py tests/test_tools_init_coverage.py -v --cov=src.cli_code.tools --cov-report=term

echo "==== Targeted coverage tests complete ===="

# Optional: Generate combined report at the end (though individual reports printed above)
# coverage combine
# coverage report -m 