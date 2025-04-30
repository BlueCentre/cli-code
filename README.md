# DEPRECATED

> **Warning**: This repository is deprecated and no longer maintained. The project has moved to [https://github.com/BlueCentre/code-agent](https://github.com/BlueCentre/code-agent).

# CLI Code

[![Python CI](https://github.com/BlueCentre/cli-code/actions/workflows/python-ci.yml/badge.svg)](https://github.com/BlueCentre/cli-code/actions/workflows/python-ci.yml)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=BlueCentre_cli-code&metric=coverage)](https://sonarcloud.io/summary/new_code?id=BlueCentre_cli-code)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=BlueCentre_cli-code&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=BlueCentre_cli-code)

An AI coding assistant for your terminal, powered by multiple LLM providers (currently Google Gemini and Ollama).

**Table of Contents**

- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- Interactive chat sessions in your terminal.
- Supports multiple model providers (Google Gemini, Ollama).
- Configurable default provider and model.
- Automatic context initialization from project files.
- Markdown rendering for chat output.
- Assistant can utilize tools for:
  - File operations (view, edit, list, grep, search)
  - Directory operations (list, tree)
  - System commands (run terminal commands)
  - Quality checks (linting, formatting)
  - Running tests (e.g., pytest)

## Installation

For detailed installation instructions, please see the [Installation Guide](docs/install.md).

**Recommended (PyPI):**
```bash
uv pip install cli-code-agent
```

**From Source:**
```bash
git clone https://github.com/BlueCentre/cli-code.git
cd cli-code
# Create a virtual environment (optional but recommended)
# uv venv
# source .venv/bin/activate

# Install in editable mode
uv pip install -e .
# For development including test dependencies, use:
# uv pip install -e '.[dev]'
```

## Setup

Configure API credentials for your desired LLM provider. See the [Installation Guide](docs/install.md) for details on setup commands and using environment variables.

**Example (Gemini):**
```bash
cli-code setup --provider=gemini YOUR_GOOGLE_API_KEY
```

Configuration is typically stored in `~/.config/cli-code/config.yaml`, but can be overridden by environment variables or a custom file path.

## Usage

Start an interactive chat session:
```bash
# Use default provider/model
cli-code

# Specify provider (uses provider's default model)
cli-code --provider=ollama

# Specify provider and model
cli-code --provider=ollama --model llama3
```

**Interactive Commands:**
- `/exit` - Exit the chat session.
- `/help` - Display help information.

## MCP Protocol Integration

The CLI Code now supports [MCP (Model Context Provider) protocol](docs/mcp-integration.md) for tool execution and integration with various LLM providers. This feature enables more standardized interactions between the model and tools.

### Using MCP Tools

MCP tools can be used programmatically or through command-line examples:

#### Example: Calculator Tool

```python
import asyncio
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService
from src.cli_code.mcp.tools.examples.calculator import CalculatorTool

# Register the calculator tool
registry = ToolRegistry()
registry.register(CalculatorTool.create())

# Create a tool service
service = ToolService(registry)

# Execute a calculation
async def calculate():
    result = await service.execute_tool("calculator", {
        "operation": "add",
        "a": 5,
        "b": 3
    })
    print(result)

# Run the async function
asyncio.run(calculate())
```

#### Example: GitHub Tool

```python
import asyncio
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService
from src.cli_code.mcp.tools.examples.github import GitHubTool

# Register the GitHub tool
registry = ToolRegistry()
registry.register(GitHubTool.create_list_repos_tool())
registry.register(GitHubTool.create_search_repos_tool())

# Create a tool service
service = ToolService(registry)

# List repositories for a user
async def list_repos():
    result = await service.execute_tool("github_list_repos", {
        "username": "microsoft"  # Optional: leave empty to list your own repos
    })
    print(result)

# Run the async function
asyncio.run(list_repos())
```

### Running Examples

Pre-built examples are included in the repository:

```bash
# Run the calculator example
python examples/mcp_calculator_example.py add 5 3

# Run the GitHub tool demo
python examples/github_tool_demo.py
```

### Creating Custom MCP Tools

You can create your own MCP tools by following this pattern:

1. Define a handler function with appropriate parameters
2. Create a Tool instance with the handler
3. Register the tool with the registry

```python
from src.cli_code.mcp.tools.models import Tool, ToolParameter

# Define an async handler
async def my_tool_handler(param1, param2):
    # Tool implementation
    return {"result": f"Processed {param1} and {param2}"}

# Create the tool
my_tool = Tool(
    name="my_tool",
    description="A custom tool example",
    parameters=[
        ToolParameter(
            name="param1",
            description="First parameter",
            type="string",
            required=True
        ),
        ToolParameter(
            name="param2",
            description="Second parameter",
            type="number",
            required=False
        )
    ],
    handler=my_tool_handler
)

# Register and use the tool
registry = ToolRegistry()
registry.register(my_tool)
service = ToolService(registry)
```

For more details on MCP protocol integration, see the [MCP Integration Documentation](docs/mcp-integration.md).

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Installation & Setup](docs/install.md)
- [Contributing Guide](docs/contributing.md)
- [Code Coverage](docs/CODE_COVERAGE.md)
- [Changelog](docs/changelog.md)
- [Architecture](docs/architecture.md)
- [Context Guidelines](docs/context.md)
- [Project Brainstorm](docs/brainstorm.md)

## Contributing

Contributions are welcome! Please see the [Contributing Guide](docs/contributing.md) for details on setting up a development environment and submitting pull requests.

## License

MIT
