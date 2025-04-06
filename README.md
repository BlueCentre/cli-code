# CLI Code

[![Python CI](https://github.com/BlueCentre/cli-code/actions/workflows/python-ci.yml/badge.svg)](https://github.com/BlueCentre/cli-code/actions/workflows/python-ci.yml)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=BlueCentre_cli-code&metric=coverage)](https://sonarcloud.io/summary/new_code?id=BlueCentre_cli-code)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=BlueCentre_cli-code&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=BlueCentre_cli-code)

An AI coding assistant for your terminal, powered by multiple LLM providers (Gemini and Ollama).

**Table of Contents**

- [Features](#features)
- [Installation](#installation)
  - [Method 1: Install from PyPI (Recommended)](#method-1-install-from-pypi-recommended)
  - [Method 2: Install from Source](#method-2-install-from-source)
- [Setup](#setup)
  - [Alternative Setup Using Environment Variables](#alternative-setup-using-environment-variables)
  - [Configuration File](#configuration-file)
- [Usage](#usage)
- [Interactive Commands](#interactive-commands)
- [Documentation](#documentation)
- [How It Works](#how-it-works)
  - [Tool Usage](#tool-usage)
- [Advanced Features](#advanced-features)
  - [Custom Context with `.rules`](#custom-context-with-rules)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

- Interactive chat sessions in your terminal
- Multiple model provider support (Google Gemini, Ollama - more planned)
- Configurable default provider and model
- Hierarchical context initialization from:
  - `.rules/*.md` files (if directory exists)
  - `README.md` file (fallback)
  - Directory listing (final fallback)
- Basic history management (prevents excessive length)
- Markdown rendering in the terminal
- Automatic tool usage by the assistant:
  - File operations (view, edit, list, grep, glob)
  - Directory operations (ls, tree, create_directory)
  - System commands (bash)
  - Quality checks (linting, formatting)
  - Test running capabilities (pytest, etc.)

## Installation

### Method 1: Install from PyPI (Recommended)

```bash
# Install directly from PyPI
pip install cli-code-agent
```

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/BlueCentre/cli-code.git
cd cli-code

# Install the package
pip install -e .
```

## Setup

Before using CLI Code, you need to set up API credentials for your desired provider:

```bash
# Set up Google API key for Gemini models
cli-code-agent setup --provider=gemini YOUR_GOOGLE_API_KEY

# OR Set up Ollama endpoint URL (if running Ollama locally or elsewhere)
# cli-code-agent setup --provider=ollama YOUR_OLLAMA_API_URL
```

### Alternative Setup Using Environment Variables

You can also configure CLI Code using environment variables, either by setting them directly or using a `.env` file:

1. Create a `.env` file in your working directory (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and uncomment/modify the settings you need:
   ```
   # API Keys and URLs
   CLI_CODE_GOOGLE_API_KEY=your_google_api_key_here
   CLI_CODE_OLLAMA_API_URL=http://localhost:11434/v1
   
   # Default Provider
   CLI_CODE_DEFAULT_PROVIDER=ollama
   
   # Default Model
   CLI_CODE_OLLAMA_DEFAULT_MODEL=llama3.2:latest
   ```

3. Run CLI Code normally, and it will automatically load the settings from your `.env` file:
   ```bash
   cli-code-agent
   ```

The environment variables take precedence over saved configuration, making this approach useful for temporary settings or project-specific configurations.

## Configuration File

CLI Code uses a configuration file located at `~/.config/cli-code-agent/config.yaml` to store settings like API keys/URLs, default providers/models, and other preferences. The file is created with defaults the first time you run the setup or the application.

You can edit this file directly. Here's an example structure:

```yaml
# ~/.config/cli-code-agent/config.yaml

google_api_key: YOUR_GEMINI_API_KEY_HERE # Or null if using environment variable
ollama_api_url: http://localhost:11434/v1 # Or null if not using Ollama/using env var

default_provider: gemini # Can be 'gemini' or 'ollama'
default_model: models/gemini-2.5-pro-exp-03-25 # Your preferred default Gemini model
ollama_default_model: llama3.2 # Your preferred default Ollama model

settings:
  max_tokens: 1000000 # Maximum token limit for models
  temperature: 0.5 # Model generation creativity (0.0 - 1.0)
  token_warning_threshold: 800000 # Display a warning if context approaches this limit
  auto_compact_threshold: 950000 # Attempt to compact history if context exceeds this
```

**Note:** Environment variables (like `CLI_CODE_GOOGLE_API_KEY`) will override the values set in this file.

## Usage

```bash
# Start an interactive session with the default provider/model
cli-code-agent

# Start a session with a specific provider (uses provider's default model)
cli-code-agent --provider=ollama

# Start a session with a specific provider and model
cli-code-agent --provider=ollama --model llama3

# Start a session with Gemini and a specific model
cli-code-agent --provider=gemini --model models/gemini-1.5-pro-latest

# Set default provider and model (example)
# cli-code-agent set-default-provider ollama
# cli-code-agent set-default-model llama3

# List available models for a specific provider
cli-code-agent list-models --provider=gemini
cli-code-agent list-models --provider=ollama
```

## Interactive Commands

During an interactive session, you can use these commands:

- `/exit` - Exit the chat session
- `/help` - Display help information

## Documentation

For more detailed information, please refer to the following documentation:

- [Installation Guide](docs/install.md) - Detailed instructions for installing and configuring CLI-Code
- [Contributing Guide](docs/contributing.md) - Guidelines for contributing to the project
- [Code Coverage Guide](CODE_COVERAGE.md) - Information on code coverage tools and practices
- [Changelog](docs/changelog.md) - History of changes and releases
- [Architecture](docs/architecture.md) - Technical overview of the system architecture
- [Context Guidelines](docs/context.md) - Guidelines for the CLI Code Assistant
- [Project Brainstorm](docs/brainstorm.md) - Original brainstorming document for the CLI tool

## How It Works

### Tool Usage

Unlike direct command-line tools, the CLI Code assistant uses tools automatically to help answer your questions. For example:

**Example Interaction:**

```
You: What python files are in the src/cli_code directory?

A:
[tool_code]
print(default_api.list_dir(relative_workspace_path='src/cli_code'))
[/tool_code]
[tool_output]
{
  "list_dir_response": {
    "results": ["Contents of directory:\n\n[file] main.py ...\n[file] config.py ...\n..."]
  }
}
[/tool_output]
Okay, I found the following Python files in `src/cli_code/`:
- `main.py`
- `config.py`
- `__init__.py`
- `utils.py`

There are also `models/` and `tools/` subdirectories containing more Python files.
```

This approach makes the interaction more natural, as you don't need to know the specific tool names or commands.

## Advanced Features

### Custom Context with `.rules`

The CLI Code assistant can be customized with project-specific context by creating a `.rules` directory in your project root. Any markdown files (`.md`) placed in this directory will be automatically loaded as context when the assistant starts in that directory.

This allows you to:
- Define project-specific guidelines
- Provide architectural constraints
- Set coding standards
- Describe the project's purpose and goals

Example: Create a file `.rules/python_standards.md` with your team's Python coding standards, and the assistant will follow those guidelines when helping with your Python code.

## Development

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/BlueCentre/cli-code.git
cd cli-code

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Run coverage analysis
./run_coverage.sh

# Run SonarCloud analysis locally (requires sonar-scanner CLI)
./run_sonar_scan.sh
```

The project uses [pytest](https://docs.pytest.org/) for testing and [SonarCloud](https://sonarcloud.io/) for code quality and coverage analysis.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT