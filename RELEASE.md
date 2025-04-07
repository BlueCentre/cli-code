# Release Process for cli-code-agent

This document outlines the steps to release new versions of the `cli-code-agent` package to PyPI.

## Prerequisites

1. GitHub repository access with permissions to push tags and create releases
2. PyPI account with permissions to publish the package

## Setup

### 1. Configure PyPI API Token

You can use either Trusted Publishing (OIDC) or a PyPI API token for publishing:

#### Option A: API Token (Simpler)

1. Generate a PyPI API token:
   - Go to https://pypi.org/manage/account/
   - Navigate to API tokens and create a new token with scope limited to the `cli-code-agent` project
   - Copy the token value (it will only be shown once)

2. Add the token to GitHub repository secrets:
   - Go to your GitHub repository → Settings → Secrets and variables → Actions
   - Create a new repository secret named `PYPI_API_TOKEN`
   - Paste the PyPI token value

#### Option B: Trusted Publishing (More Secure)

Set up Trusted Publishing between GitHub and PyPI:
   
1. On PyPI:
   - Go to your project page
   - Navigate to "Settings" → "Publishing"
   - Add a new "Pending publisher"
   - Select GitHub as the workflow
   - Enter `BlueCentre/cli-code` as the owner/repo
   - Enter `.github/workflows/python-ci.yml` as the workflow name
   - Save the publisher

2. On GitHub:
   - Create an environment named `publish` in your repository settings
   - Set the environment variable `USE_TRUSTED_PUBLISHING=true`

### 2. Creating a Release

1. Update version in `pyproject.toml`:
   ```toml
   version = "x.y.z"  # Update this line
   ```

2. Commit the version change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to x.y.z"
   ```

3. Create and push a tag:
   ```bash
   git tag -a vx.y.z -m "Release version x.y.z"
git push origin main
git push origin vx.y.z
   ```

4. Monitor the CI workflow in GitHub Actions to verify the release process completes successfully.

5. Check that the package appears on PyPI at https://pypi.org/project/cli-code-agent/

## Troubleshooting

If the release fails, check:

1. GitHub Actions logs for error messages
2. Verify that the tag format is correct (should start with 'v')
3. Ensure the PyPI token has not expired
4. Verify that the package version is unique (PyPI rejects duplicate versions) 