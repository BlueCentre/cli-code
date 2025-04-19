#!/bin/bash

# This script configures branch protection rules for the main branch
# Requires GitHub CLI with admin access to the repository

# Skip checks in CI environment
if [ -z "$CI" ]; then
    # Check if GitHub CLI is installed
    if ! command -v gh &> /dev/null; then
        echo "Error: GitHub CLI (gh) is not installed. Please install it first."
        echo "Visit https://cli.github.com/ for installation instructions."
        exit 1
    fi

    # Check if the user is authenticated with GitHub CLI
    if ! gh auth status &> /dev/null; then
        echo "Error: You are not authenticated with GitHub CLI."
        echo "Please run 'gh auth login' first to authenticate."
        exit 1
    fi
else
    echo "Running in CI environment. Skipping local GitHub CLI checks."
    # In CI environments, we assume GitHub CLI is configured via GitHub Actions
fi

# Define branch protection configuration
cat > branch_protection.json << 'EOL'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["build-and-test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOL

# Apply branch protection rules
echo "Applying branch protection rules to main branch..."
if [ -n "$CI" ]; then
    echo "In CI environment - would apply branch protection with:"
    cat branch_protection.json
    rm branch_protection.json
    echo "Branch protection rules would be applied in a non-CI environment."
    exit 0
fi

if ! gh api --method PUT "repos/BlueCentre/cli-code/branches/main/protection" \
    --input branch_protection.json; then
    echo "Failed to apply branch protection rules. Ensure you have admin access to the repository."
    rm branch_protection.json
    exit 1
fi

# Clean up
rm branch_protection.json

echo "Branch protection rules applied successfully."
