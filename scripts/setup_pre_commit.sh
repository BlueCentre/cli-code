#!/bin/bash
# This script installs and sets up pre-commit hooks for the project

# Exit on error
set -e

echo "Setting up pre-commit hooks for CLI Code..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
else
    echo "pre-commit already installed. Checking for updates..."
    pip install --upgrade pre-commit
fi

# Install the pre-commit hooks
echo "Installing git hooks from .pre-commit-config.yaml..."
pre-commit install

# Run against all files (optional)
echo "Running pre-commit on all files (first time may take a while)..."
pre-commit run --all-files

echo "✅ Pre-commit hooks have been configured successfully!"
echo "Now Ruff linting and formatting will run automatically before each commit." 