# Code Coverage Guide

This document provides information on how to run and analyze code coverage for the CLI Code Agent project.

## Overview

Code coverage is a metric that helps identify which parts of your codebase are being executed during your tests. It helps in determining the effectiveness of your tests and identifying areas that need additional testing.

## Running Code Coverage

### Using the Shell Script

We provide a convenient shell script that runs the tests and generates coverage reports:

```bash
./run_coverage.sh
```

This script will:
1. Run all tests with coverage enabled
2. Generate coverage reports in XML and HTML formats
3. Display the overall coverage percentage
4. Open the HTML report (if on macOS)

### Manual Execution

If you prefer to run coverage manually, use the following commands:

```bash
# Run pytest with coverage
python -m pytest --cov=src/cli_code --cov-report=term --cov-report=xml --cov-report=html test_dir/

# To run specific test files
python -m pytest --cov=src/cli_code --cov-report=term test_dir/test_file.py
```

## Analyzing Coverage Results

### HTML Report

The HTML report provides a user-friendly interface to explore your coverage results. After running the coverage, open:

```
coverage_html/index.html
```

### Using the Coverage Analyzer

We provide a Python script to analyze coverage data and identify modules with low coverage:

```bash
./find_low_coverage.py
```

This script will:
1. Parse the coverage XML file
2. Calculate coverage for each module
3. Display a table of module coverage
4. Highlight modules that fall below the minimum threshold (60%)
5. Provide a summary of modules needing improvement

### Using SonarCloud Locally

For the most accurate analysis that matches what runs in the CI pipeline, you can use the SonarCloud scanner locally:

```bash
./run_sonar_scan.sh
```

This script will:
1. Check if you have sonar-scanner installed
2. Verify that you have set the SONAR_TOKEN environment variable
3. Run coverage if coverage.xml doesn't exist
4. Run sonar-scanner with the same configuration as the CI pipeline
5. Provide a link to view results on SonarCloud

To install the sonar-scanner CLI:
1. Follow the instructions at [SonarScanner CLI](https://docs.sonarcloud.io/advanced-setup/ci-based-analysis/sonarscanner-cli/)
2. Set up your SONAR_TOKEN environment variable:
   ```bash
   export SONAR_TOKEN=your-token
   ```

The SonarCloud analysis provides more advanced metrics beyond just coverage, including:
- Code quality issues
- Code smells
- Security vulnerabilities
- Maintainability rating
- Reliability rating
- Security rating

## Coverage Configuration

The coverage configuration is defined in two places:

1. `.coveragerc` file - Controls how coverage is measured and reported
2. `pyproject.toml` - Contains the pytest-cov configuration

Key settings include:
- `source` - Specifies which packages to measure
- `omit` - Patterns for files to exclude from coverage
- `exclude_lines` - Patterns for lines to exclude from coverage calculation
- `fail_under` - Minimum required coverage percentage

## Continuous Integration

Coverage is automatically run as part of our CI/CD pipeline. The GitHub workflow:

1. Runs tests with coverage enabled
2. Generates coverage reports
3. Uploads coverage reports as artifacts
4. Sends coverage data to SonarCloud for analysis

The pipeline uses the exact same SonarCloud configuration that you can run locally with `./run_sonar_scan.sh`.

## Improving Coverage

To improve code coverage:

1. Identify modules with low coverage using the analysis tools
2. Focus on writing tests for uncovered code paths
3. Pay special attention to error handling and edge cases
4. Use parameterized tests for code with many variations

Remember that 100% coverage doesn't guarantee bug-free code, but it does help ensure that most code paths have been executed at least once during testing.

## Minimum Coverage Threshold

The minimum acceptable coverage for this project is **60%**. This threshold is enforced in:
- The coverage configuration
- The analysis scripts
- (Optionally) The CI pipeline

## Tips for Writing Testable Code

- Keep functions small and focused on a single responsibility
- Extract complex logic into separate testable functions
- Use dependency injection to make external dependencies mockable
- Avoid global state when possible
- Use clear error handling patterns 