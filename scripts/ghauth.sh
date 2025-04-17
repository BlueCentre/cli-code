#!/bin/bash
# ghauth.sh - GitHub CLI wrapper to use the correct authentication token
# Usage: source scripts/ghauth.sh
#        ghauth <any-gh-command>

ghauth() {
  GITHUB_TOKEN="" gh "$@"
}

echo "GitHub CLI auth wrapper loaded. Use 'ghauth' instead of 'gh' for commands requiring full repo access."
