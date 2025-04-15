"""
Tests for GitHub tool implementations.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_code.mcp.tools.examples.github import (
    GitHubTool,
    github_list_repos_handler,
    github_search_repos_handler,
)
from cli_code.mcp.tools.models import Tool


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


class TestGitHubToolHandlers:
    """Tests for the GitHub tool handlers."""
    
    @pytest.mark.asyncio
    @patch("cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("cli_code.mcp.tools.examples.github._list_repos_using_gh_cli")
    async def test_list_repos_handler_with_gh_cli(self, mock_list_repos, mock_is_gh_cli_available):
        """Test listing repositories using GitHub CLI."""
        # Mock GitHub CLI availability and return value
        mock_is_gh_cli_available.return_value = True
        mock_list_repos.return_value = {
            "repositories": [
                {
                    "name": "test-repo",
                    "description": "Test repository",
                    "url": "https://github.com/user/test-repo"
                }
            ],
            "count": 1
        }
        
        # Call the handler
        result = await github_list_repos_handler(username="testuser")
        
        # Check the result
        assert result["count"] == 1
        assert len(result["repositories"]) == 1
        assert result["repositories"][0]["name"] == "test-repo"
        
        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_list_repos.assert_called_once_with("testuser")
    
    @pytest.mark.asyncio
    @patch("cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("cli_code.mcp.tools.examples.github._list_repos_using_api")
    async def test_list_repos_handler_with_api(self, mock_list_repos, mock_is_gh_cli_available):
        """Test listing repositories using GitHub API."""
        # Mock GitHub CLI unavailability and API return value
        mock_is_gh_cli_available.return_value = False
        mock_list_repos.return_value = {
            "repositories": [
                {
                    "name": "api-repo",
                    "description": "API repository",
                    "html_url": "https://github.com/user/api-repo"
                }
            ],
            "count": 1
        }
        
        # Call the handler
        result = await github_list_repos_handler()
        
        # Check the result
        assert result["count"] == 1
        assert len(result["repositories"]) == 1
        assert result["repositories"][0]["name"] == "api-repo"
        
        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_list_repos.assert_called_once_with(None)
    
    @pytest.mark.asyncio
    @patch("cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("cli_code.mcp.tools.examples.github._search_repos_using_gh_cli")
    async def test_search_repos_handler_with_gh_cli(self, mock_search_repos, mock_is_gh_cli_available):
        """Test searching repositories using GitHub CLI."""
        # Mock GitHub CLI availability and return value
        mock_is_gh_cli_available.return_value = True
        mock_search_repos.return_value = {
            "query": "test",
            "results": [
                {
                    "name": "test-repo",
                    "description": "Test repository",
                    "url": "https://github.com/user/test-repo"
                }
            ],
            "count": 1
        }
        
        # Call the handler
        result = await github_search_repos_handler(query="test", limit=5)
        
        # Check the result
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["name"] == "test-repo"
        
        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_search_repos.assert_called_once_with("test", 5)
    
    @pytest.mark.asyncio
    @patch("cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("cli_code.mcp.tools.examples.github._search_repos_using_api")
    async def test_search_repos_handler_with_api(self, mock_search_repos, mock_is_gh_cli_available):
        """Test searching repositories using GitHub API."""
        # Mock GitHub CLI unavailability and API return value
        mock_is_gh_cli_available.return_value = False
        mock_search_repos.return_value = {
            "query": "api",
            "results": [
                {
                    "name": "api-repo",
                    "description": "API repository",
                    "html_url": "https://github.com/user/api-repo"
                }
            ],
            "count": 1
        }
        
        # Call the handler
        result = await github_search_repos_handler(query="api")
        
        # Check the result
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["name"] == "api-repo"
        
        # Verify the correct methods were called
        mock_is_gh_cli_available.assert_called_once()
        mock_search_repos.assert_called_once_with("api", 10)  # Default limit 