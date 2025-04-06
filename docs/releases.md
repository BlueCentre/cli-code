# Release Process Documentation

This document outlines the process for creating and publishing new releases of the CLI Code Agent.

## Automated Publishing to PyPI

Starting with version 0.2.0, the CLI Code Agent project uses GitHub Actions to automatically publish new releases to PyPI whenever a new tag is pushed to the repository.

## How to Create a New Release

To create a new release:

1. Update the version number in `pyproject.toml`
2. Make sure all changes are committed to the main branch
3. Create and push a new version tag:

```bash
# Ensure you're on the main branch
git checkout main
git pull

# Create an annotated tag (the v prefix is required to trigger the workflow)
git tag -a v0.2.1 -m "Release v0.2.1 - brief description of changes"

# Push the tag to GitHub
git push --tags
```

4. The GitHub Actions workflow will automatically:
   - Run tests and linting
   - Build the package
   - Publish the package to PyPI (only if all tests pass)

5. Monitor the workflow progress in the GitHub Actions tab of your repository

## Setting Up PyPI Trusted Publishing (One-time Setup)

Before the automated publishing will work, you need to configure trusted publishing:

1. **Create the PyPI Environment in GitHub:**
   - Go to your GitHub repository → Settings → Environments
   - Create a new environment named "publish"
   - (Optional) Add any environment protection rules like required reviewers

2. **Set up Trusted Publishing in PyPI:**
   - Log in to your PyPI account
   - Go to your cli-code-agent project → Settings → Publishing
   - Set up a new pending publisher:
     - Select GitHub Actions as the publisher
     - Enter your GitHub organization/username and repository name
     - Enter "publish" as the environment name 
     - Save the publisher settings

## Versioning Guidelines

- Follow [Semantic Versioning](https://semver.org/):
  - MAJOR version for incompatible API changes
  - MINOR version for backward-compatible new features
  - PATCH version for backward-compatible bug fixes

- For pre-release versions, use suffixes like `-alpha.1`, `-beta.2`, etc.

### Detailed Version Bump Criteria

#### When to Bump MAJOR Version (0.x.x → 1.0.0 or 1.x.x → 2.0.0)
- Breaking changes to the API or command structure
- Significant rewrites that change how users interact with the tool
- When the API is considered stable and feature-complete for a first official release (0.x.x → 1.0.0)
- Dropping support for previously supported Python versions or dependencies
- Changes that require users to modify their usage or configuration

#### When to Bump MINOR Version (0.2.x → 0.3.0)
- Adding new commands or features that don't break existing functionality
- Adding support for new model providers or significant model upgrades
- Implementing new tools or capabilities
- Significant improvements to performance or UX that don't change existing behavior
- Adding new configuration options
- Deprecating features (but not removing them yet)

#### When to Bump PATCH Version (0.2.1 → 0.2.2)
- Bug fixes and error corrections
- Minor documentation updates
- Test coverage improvements
- Non-functional code refactoring
- Performance optimizations that don't change behavior
- Dependency version updates for security fixes

### Version Bump Checklist

Before releasing a new version:

1. Ensure all changes are documented in the changelog (`docs/changelog.md`)
2. Update version number in `pyproject.toml`
3. Make sure all tests pass and CI checks are green
4. Confirm that the README.md and documentation are up to date
5. For MAJOR or significant MINOR releases, create a GitHub Release with detailed notes

## Release Notes

When creating significant releases, consider:

1. Creating a formal GitHub Release with detailed notes:
   - Go to GitHub repository → Releases → Draft a new release
   - Select your newly created tag
   - Add a title and description with key changes
   - Include upgrade instructions if necessary

2. Updating the README.md to mention new features

## Troubleshooting Publishing Issues

If the automated publishing fails:

1. Check the GitHub Actions logs for detailed error messages
2. Verify PyPI credentials and trusted publishing setup
3. Ensure the version number in `pyproject.toml` is unique and hasn't been published before
4. Check if the package passes all tests and linting checks

## Manual Publishing (Fallback)

If you need to publish manually:

```bash
# Build the package
python -m build

# Publish to PyPI (requires PyPI credentials)
python -m twine upload dist/cli_code_agent-VERSION*
``` 