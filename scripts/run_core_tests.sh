#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

echo "Running core tests (tests that don't require API access)..."

# Run tests excluding those that require external API calls
python -m pytest \
  tests/tools/test_tree_tool_edge_cases.py \
  tests/models/test_ollama_model_context.py \
  tests/test_basic_functions.py \
  tests/test_config.py \
  tests/test_config_edge_cases.py::TestConfigNullHandling \
  tests/test_config_edge_cases.py::TestConfigEdgeCases \
  tests/tools/test_file_tools.py \
  tests/tools/test_directory_tools.py \
  tests/tools/test_tree_tool.py \
  tests/test_utils.py \
  tests/tools/test_tools_base.py \
  -v

exit_code=$?

# Check if tests were successful
if [ $exit_code -eq 0 ]; then
  echo -e "\n\033[32mCore tests passed! ✓\033[0m"
else
  echo -e "\n\033[31mSome core tests failed. See details above. ✗\033[0m"
fi

exit $exit_code
