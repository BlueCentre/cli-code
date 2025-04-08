#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

echo "Warning: These tests require API access and may hang if services are not available."
echo "Running API-dependent tests with a 10-second timeout per test..."

# Run API-dependent tests
python -m pytest \
  test_dir/test_ollama_model.py \
  test_dir/test_gemini_model.py \
  test_dir/test_model_integration.py \
  -v \
  --timeout=10

exit_code=$?

# Check if tests were successful
if [ $exit_code -eq 0 ]; then
  echo -e "\n\033[32mAPI tests passed! ✓\033[0m"
else
  echo -e "\n\033[31mSome API tests failed or timed out. See details above. ✗\033[0m"
  echo "Note: Failures may be due to API services being unavailable."
fi

exit $exit_code 