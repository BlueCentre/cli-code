#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

echo "Running core tests (tests that don't require API access)..."

# Run tests excluding those that require external API calls
python -m pytest \
  test_dir/test_tree_tool_edge_cases.py \
  test_dir/test_ollama_model_context.py \
  test_dir/test_basic_functions.py \
  test_dir/test_config.py \
  test_dir/test_config_edge_cases.py::TestConfigNullHandling \
  test_dir/test_config_edge_cases.py::TestConfigEdgeCases \
  test_dir/test_file_tools.py \
  test_dir/test_directory_tools.py \
  test_dir/test_tree_tool.py \
  test_dir/test_utils.py \
  test_dir/test_tools_base.py \
  -v

exit_code=$?

# Check if tests were successful
if [ $exit_code -eq 0 ]; then
  echo -e "\n\033[32mCore tests passed! ✓\033[0m"
else
  echo -e "\n\033[31mSome core tests failed. See details above. ✗\033[0m"
fi

exit $exit_code 