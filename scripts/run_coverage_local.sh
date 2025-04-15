#!/bin/bash
# Script optimized for pre-commit hook usage to check coverage quickly
# Ensures complete consistency with CI pipeline coverage checks

set -e  # Exit on error

# Use colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting coverage check (using same configuration as CI)...${NC}"

# Create required directories
mkdir -p coverage_html test_logs

# Install toml if not available
python -c "import toml" 2>/dev/null || pip install toml >/dev/null

# Extract the coverage threshold from pyproject.toml
MIN_COVERAGE=$(python -c "import toml; data = toml.load('pyproject.toml'); print(data.get('tool', {}).get('coverage', {}).get('report', {}).get('fail_under', 60))")
echo -e "Using minimum coverage threshold: ${GREEN}${MIN_COVERAGE}%${NC}"

# Clean up any pycache files to avoid import conflicts (like in CI)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="test_logs/test_summary_${TIMESTAMP}.log"
echo "Test run started at $(date)" > "$SUMMARY_LOG"

# Run tests with coverage enabled - identical flags to CI script
echo -e "${YELLOW}Running tests with coverage...${NC}"
python -m pytest \
  --cov=src/cli_code \
  --cov-report=xml:coverage.xml \
  --cov-report=html:coverage_html \
  --cov-report=term-missing \
  tests/ > test_logs/pytest_output.log 2>&1

EXIT_CODE=$?

# Check if coverage.xml exists
if [ ! -f coverage.xml ]; then
  echo -e "${RED}No coverage report was generated! Tests may have failed.${NC}"
  echo -e "${YELLOW}See test output in test_logs/pytest_output.log${NC}"
  cat test_logs/pytest_output.log
  exit 1
fi

# Extract coverage percentage using XML report
COVERAGE_PCT=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); print(float(root.attrib['line-rate']) * 100)")

# Format coverage to 2 decimal places
COVERAGE_PCT=$(printf "%.2f" $COVERAGE_PCT)

echo -e "${YELLOW}Coverage: ${GREEN}${COVERAGE_PCT}%${NC}"
echo -e "Minimum required: ${YELLOW}${MIN_COVERAGE}%${NC}"

# Summary in log file (like in CI)
echo "=======================================" | tee -a "$SUMMARY_LOG"
echo "Test Run Summary:" | tee -a "$SUMMARY_LOG"
echo "- Coverage: ${COVERAGE_PCT}%" | tee -a "$SUMMARY_LOG"
echo "- Threshold: ${MIN_COVERAGE}%" | tee -a "$SUMMARY_LOG"
echo "- Exit Code: ${EXIT_CODE}" | tee -a "$SUMMARY_LOG"
echo "Test run finished at $(date)" | tee -a "$SUMMARY_LOG"
echo "=======================================" | tee -a "$SUMMARY_LOG"

# Compare coverage percentage to minimum threshold
# Use bc if available for floating point comparison, fallback to python
if command -v bc >/dev/null 2>&1; then
  if (( $(echo "$COVERAGE_PCT < $MIN_COVERAGE" | bc -l) )); then
    echo -e "${RED}Coverage is below the minimum threshold!${NC}"
    echo -e "${YELLOW}See detailed report in coverage_html/index.html${NC}"
    exit 1
  else
    echo -e "${GREEN}Coverage meets or exceeds the minimum threshold!${NC}"
    echo -e "${YELLOW}See detailed report in coverage_html/index.html${NC}"
    exit 0
  fi
else
  # Fallback to python if bc is not available
  if python -c "exit(0 if float($COVERAGE_PCT) >= float($MIN_COVERAGE) else 1)"; then
    echo -e "${GREEN}Coverage meets or exceeds the minimum threshold!${NC}"
    echo -e "${YELLOW}See detailed report in coverage_html/index.html${NC}"
    exit 0
  else
    echo -e "${RED}Coverage is below the minimum threshold!${NC}"
    echo -e "${YELLOW}See detailed report in coverage_html/index.html${NC}"
    exit 1
  fi
fi 