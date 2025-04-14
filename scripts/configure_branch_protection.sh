#!/bin/bash

# This script configures branch protection rules for the main branch
# Requires GitHub CLI with admin access to the repository

# Define branch protection configuration
cat > branch_protection.json << 'EOL'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["build-and-test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": null
}
EOL

# Apply branch protection rules
echo "Applying branch protection rules to main branch..."
if ! command -v gh &> /dev/null
then
  echo "GitHub CLI is not installed. Please install it and configure it properly."
  exit 1
fi

gh api --method PUT "repos/BlueCentre/cli-code/branches/main/protection" \
  --input branch_protection.json || { echo "Failed to apply branch protection rules: $?"; rm branch_protection.json; exit 1; }

# Clean up
rm branch_protection.json

echo "Branch protection rules applied successfully." 