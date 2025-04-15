"""
Tests for GitHub tool implementations.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli_code.mcp.tools.examples.github import (
    GitHubTool,
    _is_gh_cli_available,
    _list_repos_using_api,
    _list_repos_using_gh_cli,
    _search_repos_using_api,
    _search_repos_using_gh_cli,
    github_list_repos_handler,
    github_search_repos_handler,
    github_handler,
)
from src.cli_code.mcp.tools.models import Tool


class TestGitHubTool(unittest.TestCase):
    """Tests for the GitHub tool."""

    def test_create_list_repos_tool(self):
        """Test creating a list repositories tool."""
        tool = GitHubTool.create_list_repos_tool()

        # Check tool properties
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "github_list_repos")
        self.assertIn("Lists GitHub repositories", tool.description)

        # Check parameters
        self.assertEqual(len(tool.parameters), 1)
        self.assertEqual(tool.parameters[0].name, "username")
        self.assertFalse(tool.parameters[0].required)

    def test_create_search_repos_tool(self):
        """Test creating a search repositories tool."""
        tool = GitHubTool.create_search_repos_tool()

        # Check tool properties
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "github_search_repos")
        self.assertIn("Searches for GitHub repositories", tool.description)

        # Check parameters
        self.assertEqual(len(tool.parameters), 2)
        self.assertEqual(tool.parameters[0].name, "query")
        self.assertTrue(tool.parameters[0].required)
        self.assertEqual(tool.parameters[1].name, "limit")
        self.assertFalse(tool.parameters[1].required)

    def test_create(self):
        """Test creating a GitHubTool instance."""
        github_tool = GitHubTool()
        
        # Verify the tool's properties
        self.assertEqual(github_tool.name, "github")
        self.assertEqual(github_tool.description, "Search and retrieve information from GitHub repositories.")
        
        # Verify parameters
        params = github_tool.parameters
        param_names = [p.name for p in params]
        self.assertIn("operation", param_names)
        
        operation_param = next(p for p in params if p.name == "operation")
        self.assertTrue(operation_param.required)
        self.assertEqual(operation_param.type, "string")
        self.assertIn("search_repositories", operation_param.enum)
        self.assertIn("get_repository", operation_param.enum)
        
        # Verify handler function
        self.assertEqual(github_tool.handler, github_handler)

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test the execution of the GitHubTool."""
        github_tool = GitHubTool()
        
        # Mock the handler function
        mock_result = {
            "operation": "search_repositories",
            "repositories": [{"name": "test-repo", "url": "https://github.com/user/test-repo"}],
            "total_count": 1
        }
        
        with patch('cli_code.mcp.tools.examples.github.github_handler', AsyncMock(return_value=mock_result)):
            # Execute the tool
            parameters = {"operation": "search_repositories", "query": "test"}
            result = await github_tool.execute(parameters)
            
            # Verify the result
            self.assertEqual(result, mock_result)


class TestGitHubHelpers:
    """Tests for GitHub tool helper functions."""

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    def test_is_gh_cli_available_success(self, mock_run):
        """Test checking if GitHub CLI is available (success case)."""
        # Mock successful subprocess run
        mock_run.return_value = MagicMock(returncode=0)

        # Call the function
        result = _is_gh_cli_available()

        # Check that it returned True and subprocess was called correctly
        assert result is True
        mock_run.assert_called_once_with(["gh", "--version"], capture_output=True, text=True, check=False)

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    def test_is_gh_cli_available_failure(self, mock_run):
        """Test checking if GitHub CLI is available (failure case)."""
        # Mock failed subprocess run
        mock_run.return_value = MagicMock(returncode=1)

        # Call the function
        result = _is_gh_cli_available()

        # Check that it returned False and subprocess was called correctly
        assert result is False
        mock_run.assert_called_once_with(["gh", "--version"], capture_output=True, text=True, check=False)

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    def test_is_gh_cli_available_exception(self, mock_run):
        """Test checking if GitHub CLI is available (exception case)."""
        # Mock subprocess raising an exception
        mock_run.side_effect = FileNotFoundError("Command not found")

        # Call the function
        result = _is_gh_cli_available()

        # Check that it returned False
        assert result is False
        mock_run.assert_called_once_with(["gh", "--version"], capture_output=True, text=True, check=False)

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    @patch("src.cli_code.mcp.tools.examples.github.os.environ.copy")
    async def test_list_repos_using_gh_cli_no_username(self, mock_env_copy, mock_run):
        """Test listing repositories using GitHub CLI without username."""
        # Mock environment and subprocess
        mock_env_copy.return_value = {"PATH": "/usr/bin"}
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                [{"name": "repo1", "description": "Description 1", "url": "https://github.com/user/repo1"}]
            ),
            returncode=0,
        )

        # Call the function
        result = await _list_repos_using_gh_cli()

        # Check the result
        assert "repositories" in result
        assert len(result["repositories"]) == 1
        assert result["repositories"][0]["name"] == "repo1"
        assert result["count"] == 1

        # Check that subprocess was called correctly
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[:3] == ["gh", "repo", "list"]
        assert "--json" in cmd_args
        assert "--limit" in cmd_args

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    @patch("src.cli_code.mcp.tools.examples.github.os.environ.copy")
    async def test_list_repos_using_gh_cli_with_username(self, mock_env_copy, mock_run):
        """Test listing repositories using GitHub CLI with username."""
        # Mock environment and subprocess
        mock_env_copy.return_value = {"PATH": "/usr/bin"}
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                [{"name": "repo1", "description": "Description 1", "url": "https://github.com/testuser/repo1"}]
            ),
            returncode=0,
        )

        # Call the function
        result = await _list_repos_using_gh_cli("testuser")

        # Check the result
        assert "repositories" in result
        assert len(result["repositories"]) == 1
        assert result["repositories"][0]["name"] == "repo1"
        assert result["count"] == 1

        # Check that subprocess was called correctly
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[:4] == ["gh", "repo", "list", "testuser"]

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession.get")
    @patch("src.cli_code.mcp.tools.examples.github.os.environ.get")
    async def test_list_repos_using_api_no_username(self, mock_env_get, mock_session_get):
        """Test listing repositories using GitHub API without username."""
        # Mock environment and API response
        mock_env_get.return_value = "fake_token"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [
            {"name": "repo1", "description": "Description 1", "html_url": "https://github.com/user/repo1"}
        ]
        mock_session_get.return_value.__aenter__.return_value = mock_response

        # Call the function
        result = await _list_repos_using_api()

        # Check the result
        assert "repositories" in result
        assert len(result["repositories"]) == 1
        assert result["count"] == 1

        # Check that API was called correctly
        mock_session_get.assert_called_once()
        assert "https://api.github.com/user/repos" in mock_session_get.call_args[0][0]

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession.get")
    @patch("src.cli_code.mcp.tools.examples.github.os.environ.get")
    async def test_list_repos_using_api_with_username(self, mock_env_get, mock_session_get):
        """Test listing repositories using GitHub API with username."""
        # Mock environment and API response
        mock_env_get.return_value = "fake_token"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [
            {"name": "repo1", "description": "Description 1", "html_url": "https://github.com/testuser/repo1"}
        ]
        mock_session_get.return_value.__aenter__.return_value = mock_response

        # Call the function
        result = await _list_repos_using_api("testuser")

        # Check the result
        assert "repositories" in result
        assert len(result["repositories"]) == 1
        assert result["count"] == 1

        # Check that API was called correctly
        mock_session_get.assert_called_once()
        assert "https://api.github.com/users/testuser/repos" in mock_session_get.call_args[0][0]

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession.get")
    @patch("src.cli_code.mcp.tools.examples.github.os.environ.get")
    async def test_list_repos_using_api_error(self, mock_env_get, mock_session_get):
        """Test listing repositories using GitHub API with error response."""
        # Mock environment and API error response
        mock_env_get.return_value = "fake_token"
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "Not Found"
        mock_session_get.return_value.__aenter__.return_value = mock_response

        # Call the function and check for exception
        with pytest.raises(ValueError, match="GitHub API error"):
            await _list_repos_using_api("nonexistent")

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    @patch("src.cli_code.mcp.tools.examples.github.os.environ.copy")
    async def test_search_repos_using_gh_cli(self, mock_env_copy, mock_run):
        """Test searching repositories using GitHub CLI."""
        # Mock environment and subprocess
        mock_env_copy.return_value = {"PATH": "/usr/bin"}
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                [{"name": "test-repo", "description": "Test repo", "url": "https://github.com/user/test-repo"}]
            ),
            returncode=0,
        )

        # Call the function
        result = await _search_repos_using_gh_cli("test", 5)

        # Check the result
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["name"] == "test-repo"
        assert result["count"] == 1
        assert result["query"] == "test"

        # Check that subprocess was called correctly
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[:3] == ["gh", "search", "repos"]
        assert cmd_args[3] == "test"
        assert "--limit" in cmd_args
        assert "5" in cmd_args

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession.get")
    @patch("src.cli_code.mcp.tools.examples.github.os.environ.get")
    async def test_search_repos_using_api(self, mock_env_get, mock_session_get):
        """Test searching repositories using GitHub API."""
        # Mock environment and API response
        mock_env_get.return_value = "fake_token"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "items": [
                {"name": "test-repo", "description": "Test repo", "html_url": "https://github.com/user/test-repo"}
            ]
        }
        mock_session_get.return_value.__aenter__.return_value = mock_response

        # Call the function
        result = await _search_repos_using_api("test", 5)

        # Check the result
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["count"] == 1
        assert result["query"] == "test"

        # Check that API was called correctly
        mock_session_get.assert_called_once()
        api_url = mock_session_get.call_args[0][0]
        assert "https://api.github.com/search/repositories" in api_url
        assert "q=test" in api_url
        assert "per_page=5" in api_url


class TestGitHubToolHandlers:
    """Tests for the GitHub tool handlers."""

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_gh_cli")
    async def test_list_repos_handler_with_gh_cli(self, mock_list_repos, mock_is_gh_cli_available):
        """Test listing repositories using GitHub CLI."""
        # Mock GitHub CLI availability and return value
        mock_is_gh_cli_available.return_value = True
        mock_list_repos.return_value = {
            "repositories": [
                {"name": "test-repo", "description": "Test repository", "url": "https://github.com/user/test-repo"}
            ],
            "count": 1,
        }

        # Call the handler
        result = await github_list_repos_handler({"username": "testuser"})

        # Check the result
        assert result["count"] == 1
        assert len(result["repositories"]) == 1
        assert result["repositories"][0]["name"] == "test-repo"

        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_list_repos.assert_called_once_with("testuser")

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_api")
    async def test_list_repos_handler_with_api(self, mock_list_repos, mock_is_gh_cli_available):
        """Test listing repositories using GitHub API."""
        # Mock GitHub CLI unavailability and API return value
        mock_is_gh_cli_available.return_value = False
        mock_list_repos.return_value = {
            "repositories": [
                {"name": "api-repo", "description": "API repository", "html_url": "https://github.com/user/api-repo"}
            ],
            "count": 1,
        }

        # Call the handler
        result = await github_list_repos_handler({})

        # Check the result
        assert result["count"] == 1
        assert len(result["repositories"]) == 1
        assert result["repositories"][0]["name"] == "api-repo"

        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_list_repos.assert_called_once_with(None)

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_gh_cli")
    async def test_search_repos_handler_with_gh_cli(self, mock_search_repos, mock_is_gh_cli_available):
        """Test searching repositories using GitHub CLI."""
        # Mock GitHub CLI availability and return value
        mock_is_gh_cli_available.return_value = True
        mock_search_repos.return_value = {
            "query": "test",
            "results": [
                {"name": "test-repo", "description": "Test repository", "url": "https://github.com/user/test-repo"}
            ],
            "count": 1,
        }

        # Call the handler
        result = await github_search_repos_handler({"query": "test", "limit": 5})

        # Check the result
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["name"] == "test-repo"

        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_search_repos.assert_called_once_with("test", 5)

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_api")
    async def test_search_repos_handler_with_api(self, mock_search_repos, mock_is_gh_cli_available):
        """Test searching repositories using GitHub API."""
        # Mock GitHub CLI unavailability and API return value
        mock_is_gh_cli_available.return_value = False
        mock_search_repos.return_value = {
            "query": "api",
            "results": [
                {"name": "api-repo", "description": "API repository", "html_url": "https://github.com/user/api-repo"}
            ],
            "count": 1,
        }

        # Call the handler
        result = await github_search_repos_handler({"query": "api"})

        # Check the result
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["name"] == "api-repo"

        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_search_repos.assert_called_once_with("api", 10)  # Default limit

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_gh_cli")
    async def test_list_repos_handler_error(self, mock_list_repos, mock_is_gh_cli_available):
        """Test handling errors in the list repos handler."""
        # Mock GitHub CLI availability but raise an exception in the implementation
        mock_is_gh_cli_available.return_value = True
        mock_list_repos.side_effect = Exception("Test error")

        # Call the handler and verify it raises the correct exception
        with pytest.raises(ValueError, match="Failed to list GitHub repositories"):
            await github_list_repos_handler({})

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_gh_cli")
    async def test_search_repos_handler_error(self, mock_search_repos, mock_is_gh_cli_available):
        """Test handling errors in the search repos handler."""
        # Mock GitHub CLI availability but raise an exception in the implementation
        mock_is_gh_cli_available.return_value = True
        mock_search_repos.side_effect = Exception("Test error")

        # Call the handler and verify it raises the correct exception
        with pytest.raises(ValueError, match="Failed to search GitHub repositories"):
            await github_search_repos_handler({"query": "test"})


class TestGitHubHandler:
    @pytest.mark.asyncio
    async def test_search_repositories_success(self):
        """Test successful repository search."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "items": [
                {"name": "repo1", "description": "Test repo 1", "html_url": "https://github.com/user/repo1"},
                {"name": "repo2", "description": "Test repo 2", "html_url": "https://github.com/user/repo2"}
            ],
            "total_count": 2
        })
        
        # Mock the aiohttp.ClientSession
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=mock_response)
        
        # Call the handler
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await github_handler({"operation": "search_repositories", "query": "test repo"})
        
        # Verify the result
        assert result["operation"] == "search_repositories"
        assert len(result["repositories"]) == 2
        assert result["repositories"][0]["name"] == "repo1"
        assert result["repositories"][1]["name"] == "repo2"
        assert result["total_count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_repository_success(self):
        """Test successful repository fetch."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "name": "test-repo",
            "description": "A test repository",
            "html_url": "https://github.com/user/test-repo",
            "owner": {"login": "user"},
            "stargazers_count": 10,
            "forks_count": 5
        })
        
        # Mock the aiohttp.ClientSession
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=mock_response)
        
        # Call the handler
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await github_handler({"operation": "get_repository", "owner": "user", "repo": "test-repo"})
        
        # Verify the result
        assert result["operation"] == "get_repository"
        assert result["repository"]["name"] == "test-repo"
        assert result["repository"]["description"] == "A test repository"
        assert result["repository"]["stars"] == 10
        assert result["repository"]["forks"] == 5
    
    @pytest.mark.asyncio
    async def test_missing_operation(self):
        """Test missing operation parameter."""
        with pytest.raises(ValueError, match="Missing required parameter: operation"):
            await github_handler({})
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test invalid operation parameter."""
        with pytest.raises(ValueError, match="Invalid operation: invalid_op"):
            await github_handler({"operation": "invalid_op"})
    
    @pytest.mark.asyncio
    async def test_search_repositories_missing_query(self):
        """Test search_repositories with missing query."""
        with pytest.raises(ValueError, match="Missing required parameter: query"):
            await github_handler({"operation": "search_repositories"})
    
    @pytest.mark.asyncio
    async def test_get_repository_missing_parameters(self):
        """Test get_repository with missing parameters."""
        with pytest.raises(ValueError, match="Missing required parameter: owner"):
            await github_handler({"operation": "get_repository"})
        
        with pytest.raises(ValueError, match="Missing required parameter: repo"):
            await github_handler({"operation": "get_repository", "owner": "user"})
    
    @pytest.mark.asyncio
    async def test_github_api_error(self):
        """Test handling of GitHub API errors."""
        # Create mock response for error
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={"message": "Not Found"})
        
        # Mock the aiohttp.ClientSession
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=mock_response)
        
        # Call the handler
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(Exception, match="GitHub API error: Not Found"):
                await github_handler({"operation": "get_repository", "owner": "user", "repo": "non-existent"})
