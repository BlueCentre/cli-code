#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

echo "Running all tests with a 10-second timeout per test..."
echo "Note: This will run both core tests and API-dependent tests."
echo "API-dependent tests may fail if services are not available."

# Run all tests with timeout
python -m pytest -v --timeout=10

exit_code=$?

# Check if tests were successful
if [ $exit_code -eq 0 ]; then
  echo -e "\n\033[32mAll tests passed! ✓\033[0m"
else
  echo -e "\n\033[31mSome tests failed or timed out. See details above. ✗\033[0m"
  echo "Note: API-dependent test failures may be due to services being unavailable."
  echo "Try running core tests only: ./run_core_tests.sh"
fi

exit $exit_code 