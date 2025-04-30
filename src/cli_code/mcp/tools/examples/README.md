# MCP Protocol Example Tools

This directory contains example tool implementations for the MCP protocol. These tools demonstrate how to create and use tools with the MCP protocol framework.

## Available Tools

### Calculator Tool

A simple calculator tool that can perform basic arithmetic operations (add, subtract, multiply, divide).

Usage example:
```python
from cli_code.mcp.tools.examples import CalculatorTool
from cli_code.mcp.tools.registry import ToolRegistry

# Create a registry
registry = ToolRegistry()

# Register the calculator tool
registry.register(CalculatorTool.create())

# Execute the calculator tool (using the ToolExecutor or ToolService)
result = await executor.execute("calculator", {
    "operation": "add",
    "operands": [1, 2, 3]
})  # Returns 6
```

### Weather Tool

A tool for retrieving weather information for a specified location.

Usage example:
```python
from cli_code.mcp.tools.examples import WeatherTool
from cli_code.mcp.tools.registry import ToolRegistry

# Create a registry
registry = ToolRegistry()

# Register the weather tool
registry.register(WeatherTool.create())

# Execute the weather tool (using the ToolExecutor or ToolService)
result = await executor.execute("weather", {
    "location": "New York"
})  # Returns weather data for New York
```

### GitHub Tool

A tool for interacting with GitHub repositories. It provides functionality for listing and searching repositories.

The GitHub tool includes two separate tools:
1. `github_list_repos` - Lists repositories for a user
2. `github_search_repos` - Searches for repositories matching a query

The tool first tries to use the GitHub CLI (`gh`) if available, which handles authentication through the system keyring. If the GitHub CLI is not available, it falls back to using the GitHub API with a token from the `GITHUB_TOKEN` environment variable.

Usage example:
```python
from cli_code.mcp.tools.examples import GitHubTool
from cli_code.mcp.tools.registry import ToolRegistry

# Create a registry
registry = ToolRegistry()

# Register both GitHub tools
registry.register(GitHubTool.create_list_repos_tool())
registry.register(GitHubTool.create_search_repos_tool())

# List repositories for the authenticated user
result = await executor.execute("github_list_repos", {})

# List repositories for a specific user
result = await executor.execute("github_list_repos", {
    "username": "microsoft"
})

# Search for repositories
result = await executor.execute("github_search_repos", {
    "query": "machine learning",
    "limit": 5
})
```

## Authentication Notes

### GitHub Tool Authentication

The GitHub tool supports two authentication methods:

1. **GitHub CLI (recommended)**: If the GitHub CLI (`gh`) is installed and configured, the tool will use it for authentication. This method uses the token stored in the system keyring.

   When using GitHub CLI, the tool implements a workaround for token scope issues by temporarily unsetting the `GITHUB_TOKEN` environment variable during execution.

2. **Environment Variable**: If the GitHub CLI is not available, the tool falls back to using the `GITHUB_TOKEN` environment variable. Make sure to set this variable with a valid GitHub personal access token before using the tool.

## Demo Script

For a complete example of using these tools, see the `examples/github_tool_demo.py` script in the repository root.
