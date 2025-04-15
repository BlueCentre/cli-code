#!/usr/bin/env python3
"""
Demonstration script for GitHub tool using MCP protocol.

This script showcases how to register and use the GitHub tool
for searching and listing repositories.
"""

import asyncio
import json
import logging
import os
import sys

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import from src directly 
from src.cli_code.mcp.tools.examples.github import GitHubTool
from src.cli_code.mcp.tools.registry import ToolRegistry
from src.cli_code.mcp.tools.service import ToolService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Run the GitHub tool demonstration."""
    # Create a tool registry
    registry = ToolRegistry()

    # Register the GitHub tools
    list_repos_tool = GitHubTool.create_list_repos_tool()
    search_repos_tool = GitHubTool.create_search_repos_tool()

    registry.register(list_repos_tool)
    registry.register(search_repos_tool)

    # Create a tool service
    service = ToolService(registry)

    # Show available tools
    print("Available tools:")
    for tool_name in registry.list_tools():
        print(f"- {tool_name}")
    print()

    # Execute the list repositories tool (without a username to list authenticated user's repos)
    print("\nListing repositories for authenticated user:")
    try:
        result = await service.execute_tool("github_list_repos", {})
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error listing repositories: {e}")

    # Execute the search repositories tool
    search_query = "python-tutorial"
    print(f"\nSearching for repositories matching '{search_query}':")
    try:
        result = await service.execute_tool("github_search_repos", {
            "query": search_query,
            "limit": 5
        })
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error searching repositories: {e}")

    # Execute search with a specific username
    username = "microsoft"
    print(f"\nListing repositories for user '{username}':")
    try:
        result = await service.execute_tool("github_list_repos", {
            "username": username
        })
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error listing repositories for user: {e}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
