## Overview
This PR fixes the code coverage regression by updating the import statements in test files and using improved test versions.

## Changes
- Updated import statements in test files to use direct imports from `src.cli_code` instead of `cli_code` to ensure proper coverage tracking
- Updated test scripts to use the improved test files from the `test_dir/improved` directory
- Added BaseTool tests in coverage scripts to improve coverage of the base tool class
- Fixed failing assertions in the Gemini model tests
- Updated tools coverage script to include all the necessary tool tests

## Test Results
- Tools coverage increased to 95.26% overall
- Individual components show excellent coverage:
  - 100% coverage for directory_tools, quality_tools, task_complete_tool, and test_runner
  - 98.65% coverage for summarizer_tool
  - 96.70% coverage for tree_tool
  - 89.83% coverage for file_tools
  - 87.50% coverage for base tool class

## Why It's Needed
The code coverage had regressed due to import paths not being correctly set for coverage tracking. These changes restore and improve the coverage levels while ensuring all tests pass reliably.

Fixes the coverage regression issues previously identified. 