#!/bin/bash

# Script to run test coverage and generate reports

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running test coverage for cli-code...${NC}"
echo ""

# Check for virtual environment and activate it if found
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Ensure pytest-cov is installed
echo -e "${YELLOW}Checking for pytest-cov...${NC}"
pip list | grep -q pytest-cov
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing pytest-cov...${NC}"
    pip install pytest-cov
fi

# Clean up previous coverage data
rm -rf .coverage coverage.xml coverage_html/

# Run pytest with coverage
echo -e "${YELLOW}Running pytest with coverage...${NC}"
python -m pytest --cov=src/cli_code --cov-report=term --cov-report=xml --cov-report=html --verbose test_dir/ "$@"

# Check if the tests were successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Tests completed successfully!${NC}"
else
    echo -e "\n${RED}Tests failed!${NC}"
fi

# Get the coverage percentage
if [ -f coverage.xml ]; then
    COVERAGE=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); print(f\"{float(root.attrib['line-rate'])*100:.2f}\")")
    echo -e "\n${YELLOW}Overall coverage: ${GREEN}${COVERAGE}%${NC}"
    
    # Check if coverage meets the minimum threshold (60%)
    if (( $(echo "$COVERAGE < 60" | bc -l) )); then
        echo -e "${RED}Coverage is below the minimum threshold of 60%!${NC}"
    else
        echo -e "${GREEN}Coverage meets or exceeds the minimum threshold of 60%!${NC}"
    fi
    
    echo -e "\n${YELLOW}Coverage reports generated:${NC}"
    echo -e "XML Report: ${GREEN}coverage.xml${NC}"
    echo -e "HTML Report: ${GREEN}coverage_html/index.html${NC}"
    
    # Open HTML report if on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "\n${YELLOW}Opening HTML report...${NC}"
        open coverage_html/index.html
    else
        echo -e "\n${YELLOW}To view the HTML report, open:${NC} ${GREEN}coverage_html/index.html${NC} in your browser."
    fi
else
    echo -e "\n${RED}No coverage report generated!${NC}"
fi 