# CLI Code Assistant Guidelines

## Core Principles
1. The assistant is primarily designed to help with coding tasks, focusing on intelligent file manipulation and code generation.
2. Prefer to use established patterns and conventions in the codebase.
3. Always provide clear explanations and context when making recommendations.

## Interaction Style
* Be concise and specific in responses
* Focus on technical accuracy and correctness
* Provide actionable code suggestions and examples

## Project Structure
The CLI code assistant is organized as a Python package with standard project structure:

* `src/cli_code/` - Main package code
  * `models/` - LLM interfaces (Gemini, Ollama)
  * `tools/` - CLI tools for file operations, grepping, etc.
  * `utils/` - Shared utility functions
* `.github/` - CI/CD workflows and GitHub configurations
* `docs/` - Documentation files

## Code Style
Follow these style guidelines when generating or modifying code:

* Use consistent naming conventions (snake_case for variables, functions; PascalCase for classes)
* Include type annotations for function parameters and return types
* Document functions with docstrings
* Keep functions focused on a single responsibility
