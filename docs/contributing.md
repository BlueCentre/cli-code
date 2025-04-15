# Contributing to CLI-Code

Thank you for your interest in contributing to CLI-Code! This document outlines the development workflow and best practices to follow when making changes.

## Development Workflow

### 1. Analysis and Planning
- **Analyze the issue/feature**: Understand the requirements thoroughly
- **Run local SonarCloud scan**: Get a baseline of current code quality and coverage
  ```bash
  # Generate coverage report
  pytest --cov=src tests --cov-report=xml

  # Run local SonarCloud scan
  sonar-scanner -Dsonar.login=YOUR_SONARCLOUD_TOKEN
  ```
- **Review current SonarCloud metrics**: Identify code smells, bugs, and areas with low coverage

### 2. Plan Implementation Steps
- Break down the solution into logical, manageable steps
- Document the plan, considering all edge cases
- Consider how the changes might affect existing functionality

### 3. Implementation
- Execute the plan one step at a time
- Follow the project's code style and conventions
- Keep changes focused and avoid scope creep
- Regularly commit your changes with descriptive messages

### 4. Testing
- Add or update tests to ensure changes are properly covered
- Ensure overall code coverage does not decrease
- Run the test suite frequently during development:
  ```bash
  pytest --cov=src tests
  ```

### 5. Verification
- Perform end-to-end testing with real usage scenarios
- Get user feedback when applicable
- Run a final local SonarCloud scan to verify quality improvements:
  ```bash
  # Generate final coverage report
  pytest --cov=src tests --cov-report=xml

  # Run local SonarCloud scan
  sonar-scanner -Dsonar.login=YOUR_SONARCLOUD_TOKEN
  ```

### 6. Documentation
- Update relevant documentation
- Add comments to complex sections of code
- Consider updating README.md if user-facing changes are made

### 7. Commit Preparation
- Prepare a detailed commit description
- Write a clear, concise commit message summary
- Reference any related issues in your commit message

### 8. Review and Submit
- Review your changes one final time
- Only push completed, well-tested changes
- Submit pull request if working in fork/branch model

## Additional Guidelines

### Code Quality
- Follow the project's code style (enforced by ruff)
- Address SonarCloud issues proactively
- Document public functions and methods with docstrings

### Testing Standards
- Aim for comprehensive test coverage (unit tests, integration tests)
- Test edge cases and failure scenarios
- Mock external dependencies appropriately

### Performance Considerations
- Be mindful of performance implications of your changes
- Profile code for expensive operations when necessary
- Consider memory usage for larger data processing

## SonarCloud Local Analysis

For the fastest feedback loop, run SonarCloud analysis locally before pushing changes:

1. Ensure you have sonar-scanner installed:
   ```bash
   # On macOS with Homebrew
   brew install sonar-scanner

   # On NixOS
   nix-env -iA nixpkgs.sonar-scanner-cli.out
   ```

2. Generate coverage report:
   ```bash
   pytest --cov=src tests --cov-report=xml
   ```

3. Run local scan (requires your SonarCloud token):
   ```bash
   # Option 1: Pass token directly (do not commit this command!)
   sonar-scanner -Dsonar.login=YOUR_SONARCLOUD_TOKEN

   # Option 2: Use environment variable (recommended)
   export SONAR_TOKEN=YOUR_SONARCLOUD_TOKEN
   sonar-scanner

   # Option 3: Add to ~/.bash_profile or ~/.zshrc (for convenience)
   # export SONAR_TOKEN=YOUR_SONARCLOUD_TOKEN
   ```

   You can get your SonarCloud token from your SonarCloud account under Security settings.

4. Review the results and address any issues before pushing.

This local workflow complements (but doesn't replace) the GitHub Actions workflow that runs automatically on push.

## Pre-commit Hooks

Pre-commit hooks automatically check your code before each commit to ensure it meets quality standards and passes linting. This helps prevent CI/CD pipeline failures and ensures consistent code quality.

### Setup

We've provided a script to easily install pre-commit hooks:

```bash
# Run the provided setup script
./scripts/setup_pre_commit.sh
```

This will:
1. Install pre-commit if it's not already installed
2. Set up the git hooks from our configuration
3. Run the hooks against all files to verify setup

### Available Hooks

The following checks run automatically before each commit:

1. **Ruff Linting**: Checks Python code for style issues and common problems
2. **Ruff Formatting**: Ensures consistent code formatting
3. **Additional Quality Checks**:
   - Trailing whitespace removal
   - End-of-file fixer (ensures files end with a newline)
   - YAML syntax validation
   - Prevention of large file commits

### Bypass Hooks Temporarily

If you need to bypass the hooks temporarily (not recommended), you can use:

```bash
git commit -m "Your message" --no-verify
```

### Manual Execution

You can also run the hooks manually at any time:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

This local validation ensures your code meets project standards before submission and prevents linting-related CI failures.
