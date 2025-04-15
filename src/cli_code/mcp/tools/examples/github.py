"""
GitHub tool for MCP protocol.

This module provides a GitHub tool for interacting with GitHub repositories.
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import aiohttp

from cli_code.mcp.tools.models import Tool, ToolParameter

logger = logging.getLogger(__name__)


async def github_list_repos_handler(username: Optional[str] = None) -> Dict[str, Any]:
    """
    List GitHub repositories for the authenticated user or a specified user.

    Args:
        username: Optional username to get repositories for. If not provided,
                 lists repositories for the authenticated user.

    Returns:
        Dictionary containing repository information

    Raises:
        ValueError: If the operation fails
    """
    try:
        # Use GitHub CLI if available (preferred for authentication handling)
        if _is_gh_cli_available():
            return await _list_repos_using_gh_cli(username)

        # Fall back to GitHub API with token
        return await _list_repos_using_api(username)

    except Exception as e:
        logger.exception(f"Failed to list GitHub repositories: {str(e)}")
        raise ValueError(f"Failed to list GitHub repositories: {str(e)}") from e


async def github_search_repos_handler(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search for GitHub repositories.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)

    Returns:
        Dictionary containing search results

    Raises:
        ValueError: If the search operation fails
    """
    try:
        # Use GitHub CLI if available (preferred for authentication handling)
        if _is_gh_cli_available():
            return await _search_repos_using_gh_cli(query, limit)

        # Fall back to GitHub API with token
        return await _search_repos_using_api(query, limit)

    except Exception as e:
        logger.exception(f"Failed to search GitHub repositories: {str(e)}")
        raise ValueError(f"Failed to search GitHub repositories: {str(e)}") from e


def _is_gh_cli_available() -> bool:
    """
    Check if GitHub CLI is available on the system.

    Returns:
        True if GitHub CLI is available, False otherwise
    """
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True, check=False)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


async def _list_repos_using_gh_cli(username: Optional[str] = None) -> Dict[str, Any]:
    """
    List repositories using GitHub CLI.

    Args:
        username: Optional username to list repositories for

    Returns:
        Dictionary containing repository information
    """
    cmd = ["gh", "repo", "list"]

    # Add username if specified
    if username:
        cmd.append(username)

    # Add format as JSON
    cmd.extend(["--json", "name,description,url,visibility,isPrivate,updatedAt"])
    cmd.extend(["--limit", "100"])

    # Use the workaround for GitHub CLI authentication issues
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = ""  # Unset GITHUB_TOKEN to use keyring token

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)

    repos = json.loads(result.stdout)

    return {"repositories": repos, "count": len(repos)}


async def _search_repos_using_gh_cli(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search repositories using GitHub CLI.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Dictionary containing search results
    """
    cmd = ["gh", "search", "repos"]

    # Add query
    cmd.append(query)

    # Add format as JSON and limit
    cmd.extend(["--json", "name,description,url,owner,stars,updatedAt"])
    cmd.extend(["--limit", str(limit)])

    # Use the workaround for GitHub CLI authentication issues
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = ""  # Unset GITHUB_TOKEN to use keyring token

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)

    repos = json.loads(result.stdout)

    return {"query": query, "results": repos, "count": len(repos)}


async def _list_repos_using_api(username: Optional[str] = None) -> Dict[str, Any]:
    """
    List repositories using GitHub API.

    Args:
        username: Optional username to list repositories for

    Returns:
        Dictionary containing repository information
    """
    # Get GitHub token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GitHub token not found in environment variables")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    url = "https://api.github.com/user/repos" if not username else f"https://api.github.com/users/{username}/repos"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"GitHub API error: {response.status} - {error_text}")

            repos = await response.json()

            return {"repositories": repos, "count": len(repos)}


async def _search_repos_using_api(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search repositories using GitHub API.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Dictionary containing search results
    """
    # Get GitHub token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GitHub token not found in environment variables")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    url = f"https://api.github.com/search/repositories?q={query}&per_page={limit}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"GitHub API error: {response.status} - {error_text}")

            search_results = await response.json()

            return {
                "query": query,
                "results": search_results.get("items", []),
                "count": len(search_results.get("items", [])),
            }


class GitHubTool:
    """GitHub tool for interacting with GitHub repositories."""

    @staticmethod
    def create_list_repos_tool() -> Tool:
        """
        Create a tool for listing GitHub repositories.

        Returns:
            A Tool instance for listing GitHub repositories
        """
        return Tool(
            name="github_list_repos",
            description="Lists GitHub repositories for the authenticated user or a specified user",
            parameters=[
                ToolParameter(
                    name="username",
                    description="Optional GitHub username. If not provided, lists repositories for the authenticated user",
                    type="string",
                    required=False,
                )
            ],
            handler=github_list_repos_handler,
        )

    @staticmethod
    def create_search_repos_tool() -> Tool:
        """
        Create a tool for searching GitHub repositories.

        Returns:
            A Tool instance for searching GitHub repositories
        """
        return Tool(
            name="github_search_repos",
            description="Searches for GitHub repositories",
            parameters=[
                ToolParameter(name="query", description="Search query string", type="string", required=True),
                ToolParameter(
                    name="limit",
                    description="Maximum number of results to return (default: 10)",
                    type="integer",
                    required=False,
                ),
            ],
            handler=github_search_repos_handler,
        )
