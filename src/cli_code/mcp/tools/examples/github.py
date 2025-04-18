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

from src.cli_code.mcp.tools.models import Tool, ToolParameter

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_GH_LIST_LIMIT = 100
DEFAULT_GH_SEARCH_LIMIT = 10

# GitHub API constants
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


async def github_list_repos_handler(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    List GitHub repositories for the authenticated user or a specified user.

    Args:
        parameters: Dictionary containing:
            username: Optional username to get repositories for. If not provided,
                     lists repositories for the authenticated user.

    Returns:
        Dictionary containing repository information

    Raises:
        ValueError: If the operation fails
    """
    try:
        # Extract parameters
        username = parameters.get("username")

        # Use GitHub CLI if available (preferred for authentication handling)
        if _is_gh_cli_available():
            return await _list_repos_using_gh_cli(username)

        # Fall back to GitHub API with token
        return await _list_repos_using_api(username)

    except Exception as e:
        logger.exception(f"Failed to list GitHub repositories: {str(e)}")
        raise ValueError(f"Failed to list GitHub repositories: {str(e)}") from e


async def github_search_repos_handler(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for GitHub repositories.

    Args:
        parameters: Dictionary containing:
            query: Search query string
            limit: Maximum number of results to return (default: DEFAULT_GH_SEARCH_LIMIT)

    Returns:
        Dictionary containing search results

    Raises:
        ValueError: If the search operation fails
    """
    try:
        # Extract parameters
        query = parameters.get("query")
        limit = parameters.get("limit", DEFAULT_GH_SEARCH_LIMIT)

        if not query:
            raise ValueError("Query parameter is required")

        # Use GitHub CLI if available (preferred for authentication handling)
        if _is_gh_cli_available():
            return await _search_repos_using_gh_cli(query, limit)

        # Fall back to GitHub API with token
        return await _search_repos_using_api(query, limit)

    except Exception as e:
        logger.exception(f"Failed to search GitHub repositories: {str(e)}")
        raise ValueError(f"Failed to search GitHub repositories: {str(e)}") from e


async def github_handler(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler for GitHub operations.

    Args:
        parameters: Dictionary containing:
            operation: The operation to perform (search_repositories, get_repository)
            Additional parameters depending on the operation

    Returns:
        Dictionary containing operation results

    Raises:
        ValueError: If an invalid operation is specified or if required parameters are missing
    """
    # Check for required operation parameter
    if "operation" not in parameters:
        raise ValueError("Missing required parameter: operation")

    operation = parameters["operation"]

    # Handle different operations
    if operation == "search_repositories":
        if "query" not in parameters:
            raise ValueError("Missing required parameter: query")

        query = parameters["query"]
        limit = parameters.get("limit", 10)

        # Use GitHub API to search repositories
        token = os.environ.get("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}" if token else "",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        url = f"https://api.github.com/search/repositories?q={query}&per_page={limit}"

        session = aiohttp.ClientSession()
        try:
            async with session as client:
                async with client.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.json()
                        raise Exception(f"GitHub API error: {error_text.get('message', 'Unknown error')}")

                    data = await response.json()

                    return {
                        "operation": "search_repositories",
                        "repositories": data.get("items", []),
                        "total_count": data.get("total_count", 0),
                    }
        finally:
            await session.close()

    elif operation == "get_repository":
        # Check for required parameters
        if "owner" not in parameters:
            raise ValueError("Missing required parameter: owner")
        if "repo" not in parameters:
            raise ValueError("Missing required parameter: repo")

        owner = parameters["owner"]
        repo = parameters["repo"]

        # Use GitHub API to get repository details
        token = os.environ.get("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}" if token else "",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        url = f"https://api.github.com/repos/{owner}/{repo}"

        session = aiohttp.ClientSession()
        try:
            async with session as client:
                async with client.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.json()
                        raise Exception(f"GitHub API error: {error_text.get('message', 'Unknown error')}")

                    repo_data = await response.json()

                    # Format the repository data
                    formatted_repo = {
                        "name": repo_data.get("name"),
                        "description": repo_data.get("description"),
                        "url": repo_data.get("html_url"),
                        "owner": repo_data.get("owner", {}).get("login"),
                        "stars": repo_data.get("stargazers_count"),
                        "forks": repo_data.get("forks_count"),
                        "language": repo_data.get("language"),
                        "is_private": repo_data.get("private"),
                        "created_at": repo_data.get("created_at"),
                        "updated_at": repo_data.get("updated_at"),
                    }

                    return {"operation": "get_repository", "repository": formatted_repo}
        finally:
            await session.close()

    else:
        raise ValueError(f"Invalid operation: {operation}")


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


async def _list_repos_using_gh_cli(
    username: Optional[str] = None, limit: int = DEFAULT_GH_LIST_LIMIT
) -> Dict[str, Any]:
    """
    List repositories using GitHub CLI.

    Args:
        username: Optional username to list repositories for
        limit: Maximum number of results to return

    Returns:
        Dictionary containing repository information
    """
    cmd = ["gh", "repo", "list"]

    # Add username if specified
    if username:
        cmd.append(username)

    # Add format as JSON
    cmd.extend(["--json", "name,description,url,visibility,isPrivate,updatedAt"])
    cmd.extend(["--limit", str(limit)])

    # Use the workaround for GitHub CLI authentication issues
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = ""  # Unset GITHUB_TOKEN to use keyring token

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)

    repos = json.loads(result.stdout)

    return {"repositories": repos, "count": len(repos)}


async def _search_repos_using_gh_cli(query: str, limit: int = DEFAULT_GH_SEARCH_LIMIT) -> Dict[str, Any]:
    """
    Search repositories using GitHub CLI.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Dictionary containing search results
    """
    cmd = ["gh", "search", "repos", query]
    cmd.extend(["--json", "fullName,description"])
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


def _get_github_token() -> str:
    """Helper function to get the GitHub token, raising an error if not found."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_API_TOKEN environment variable not set. Cannot use REST API features.")
    return token


async def _create_issue_using_rest(
    owner: str,
    repo: str,
    title: str,
    body: Optional[str] = None,
    labels: Optional[List[str]] = None,
    assignees: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create an issue using the GitHub REST API.

    Args:
        owner: Repository owner.
        repo: Repository name.
        title: Issue title.
        body: Issue body.
        labels: List of labels to add.
        assignees: List of assignees.

    Returns:
        The JSON response from the GitHub API.
    """
    token = _get_github_token()
    headers = GITHUB_API_HEADERS.copy()
    headers["Authorization"] = f"Bearer {token}"

    url = f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/issues"
    data: Dict[str, Any] = {"title": title}
    if body:
        data["body"] = body
    if labels:
        data["labels"] = labels
    if assignees:
        data["assignees"] = assignees

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()  # Raise exception for bad status codes
            return await response.json()


async def _search_issues_using_rest(
    query: str, limit: int = DEFAULT_GH_SEARCH_LIMIT, sort: Optional[str] = None, order: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for issues using the GitHub REST API.

    Args:
        query: The search query string.
        limit: Maximum number of results per page.
        sort: The field to sort by (e.g., 'created', 'updated').
        order: The direction to sort ('asc' or 'desc').

    Returns:
        The JSON response from the GitHub API.
    """
    token = _get_github_token()
    headers = GITHUB_API_HEADERS.copy()
    headers["Authorization"] = f"Bearer {token}"

    params: Dict[str, Any] = {"q": query, "per_page": limit}
    if sort:
        params["sort"] = sort
    if order:
        params["order"] = order

    url = f"{GITHUB_API_BASE_URL}/search/issues"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()  # Raise exception for bad status codes
            return await response.json()


class GitHubTool:
    """GitHub tool for interacting with GitHub repositories."""

    def __init__(self):
        """Initialize the GitHub tool."""
        self.name = "github"
        self.description = "Search and retrieve information from GitHub repositories."
        self.parameters = [
            ToolParameter(
                name="operation",
                description="The GitHub operation to perform",
                type="string",
                required=True,
                enum=["search_repositories", "get_repository"],
            ),
            ToolParameter(
                name="query",
                description="Search query for repositories (required for search_repositories)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="limit",
                description="Maximum number of search results (default: 10)",
                type="integer",
                required=False,
            ),
            ToolParameter(
                name="owner",
                description="Repository owner (required for get_repository)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="repo", description="Repository name (required for get_repository)", type="string", required=False
            ),
        ]
        self.handler = github_handler

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the GitHub tool with the given parameters.

        Args:
            parameters: The parameters for the GitHub operation

        Returns:
            The result of the operation
        """
        return await self.handler(parameters)

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
