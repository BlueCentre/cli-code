#!/bin/bash

# Script to run SonarCloud scanner locally (the same as in CI)

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running SonarCloud scan locally...${NC}"
echo ""

# Check if sonar-scanner is installed
if ! command -v sonar-scanner &> /dev/null; then
    echo -e "${RED}Error: sonar-scanner not found!${NC}"
    echo -e "Please install sonar-scanner first:"
    echo -e "${GREEN}https://docs.sonarcloud.io/advanced-setup/ci-based-analysis/sonarscanner-cli/${NC}"
    exit 1
fi

# Check for SONAR_TOKEN in environment
if [ -z "$SONAR_TOKEN" ]; then
    echo -e "${RED}Error: SONAR_TOKEN environment variable is not set!${NC}"
    echo -e "Please set it with your SonarCloud token:"
    echo -e "${GREEN}export SONAR_TOKEN=your-token${NC}"
    exit 1
fi

# Check if coverage file exists, if not run coverage first
if [ ! -f "coverage.xml" ]; then
    echo -e "${YELLOW}No coverage.xml found. Running coverage first...${NC}"
    ./run_coverage.sh
    
    # Check if coverage run was successful
    if [ $? -ne 0 ]; then
        echo -e "${RED}Coverage generation failed. Cannot continue.${NC}"
        exit 1
    fi
fi

# Run sonar-scanner with the same arguments as in CI
echo -e "${YELLOW}Running sonar-scanner...${NC}"
sonar-scanner \
  -Dsonar.python.coverage.reportPaths=coverage.xml \
  -Dsonar.host.url=https://sonarcloud.io \
  -Dsonar.projectKey=$(grep "sonar.projectKey" sonar-project.properties | cut -d= -f2) \
  -Dsonar.organization=$(grep "sonar.organization" sonar-project.properties | cut -d= -f2) \
  -Dsonar.sources=$(grep "sonar.sources" sonar-project.properties | cut -d= -f2) \
  -Dsonar.tests=$(grep "sonar.tests" sonar-project.properties | cut -d= -f2) \
  -Dsonar.sourceEncoding=$(grep "sonar.sourceEncoding" sonar-project.properties | cut -d= -f2) \
  -Dsonar.scm.provider=$(grep "sonar.scm.provider" sonar-project.properties | cut -d= -f2) \
  -Dsonar.coverage.jacoco.xmlReportPaths=coverage.xml

# Check if sonar-scanner was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}SonarCloud scan completed successfully!${NC}"
    
    # Extract project key for URL
    PROJECT_KEY=$(grep "sonar.projectKey" sonar-project.properties | cut -d= -f2)
    ORGANIZATION=$(grep "sonar.organization" sonar-project.properties | cut -d= -f2)
    
    echo -e "\n${YELLOW}View results at:${NC}"
    echo -e "${GREEN}https://sonarcloud.io/dashboard?id=${PROJECT_KEY}${NC}"
else
    echo -e "\n${RED}SonarCloud scan failed!${NC}"
fi 