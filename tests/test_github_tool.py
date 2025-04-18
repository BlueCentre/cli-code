"""
Tests for the GitHub example tool.
"""

import json
import os
import subprocess  # Import needed for CalledProcessError
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

from src.cli_code.mcp.tools.examples.github import (
    GitHubTool,
    # Removed github_handler as it seems unused/redundant
    _is_gh_cli_available,
    _list_repos_using_api,
    _list_repos_using_gh_cli,
    _search_repos_using_api,
    _search_repos_using_gh_cli,
    github_list_repos_handler,
    github_search_repos_handler,
)
from src.cli_code.mcp.tools.models import Tool, ToolParameter

# --- Helper Mocks ---


class MockCompletedProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def check_returncode(self):
        if self.returncode != 0:
            raise subprocess.CalledProcessError(self.returncode, "cmd", output=self.stdout, stderr=self.stderr)


class MockAiohttpClientResponse:
    def __init__(self, status=200, json_data=None, text_data="", reason="OK"):
        self.status = status
        self._json_data = json_data
        self._text_data = text_data
        self.reason = reason

    async def json(self):
        if self._json_data is None:
            # Simulate json() failing if no json_data provided
            raise aiohttp.ContentTypeError(MagicMock(), "")
        return self._json_data

    async def text(self):
        return self._text_data

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(MagicMock(), (), status=self.status, message=self.reason)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


# --- Test Cases ---


class TestGitHubHelpers(unittest.TestCase):
    """Tests for helper functions in github.py"""

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    def test_is_gh_cli_available_true(self, mock_run):
        """Test _is_gh_cli_available returns True when gh exists."""
        mock_run.return_value = MockCompletedProcess(stdout="gh version 2.0.0", returncode=0)
        self.assertTrue(_is_gh_cli_available())
        mock_run.assert_called_once_with(["gh", "--version"], capture_output=True, text=True, check=False)

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    def test_is_gh_cli_available_false_rc(self, mock_run):
        """Test _is_gh_cli_available returns False on non-zero return code."""
        mock_run.return_value = MockCompletedProcess(stderr="gh not found", returncode=1)
        self.assertFalse(_is_gh_cli_available())

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run", side_effect=FileNotFoundError)
    def test_is_gh_cli_available_false_exception(self, mock_run):
        """Test _is_gh_cli_available returns False on FileNotFoundError."""
        self.assertFalse(_is_gh_cli_available())


@pytest.mark.asyncio
class TestGitHubCliFunctions(unittest.TestCase):
    """Tests for functions interacting with gh CLI."""

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    async def test_list_repos_using_gh_cli_no_user(self, mock_run):
        repos_json = json.dumps([{"name": "repo1"}, {"name": "repo2"}])
        mock_run.return_value = MockCompletedProcess(stdout=repos_json, returncode=0)

        result = await _list_repos_using_gh_cli()

        expected_cmd = [
            "gh",
            "repo",
            "list",
            "--json",
            "name,description,url,visibility,isPrivate,updatedAt",
            "--limit",
            "100",
        ]
        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, env=unittest.mock.ANY
        )
        self.assertEqual(result, {"repositories": [{"name": "repo1"}, {"name": "repo2"}], "count": 2})
        passed_env = mock_run.call_args.kwargs["env"]
        self.assertEqual(passed_env.get("GITHUB_TOKEN"), "")

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    async def test_list_repos_using_gh_cli_with_user(self, mock_run):
        repos_json = json.dumps([{"name": "user_repo"}])
        mock_run.return_value = MockCompletedProcess(stdout=repos_json, returncode=0)

        result = await _list_repos_using_gh_cli("testuser")

        expected_cmd = [
            "gh",
            "repo",
            "list",
            "testuser",
            "--json",
            "name,description,url,visibility,isPrivate,updatedAt",
            "--limit",
            "100",
        ]
        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, env=unittest.mock.ANY
        )
        self.assertEqual(result, {"repositories": [{"name": "user_repo"}], "count": 1})
        passed_env = mock_run.call_args.kwargs["env"]
        self.assertEqual(passed_env.get("GITHUB_TOKEN"), "")

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    async def test_search_repos_using_gh_cli(self, mock_run):
        search_json = json.dumps([{"name": "found_repo"}])
        mock_run.return_value = MockCompletedProcess(stdout=search_json, returncode=0)

        result = await _search_repos_using_gh_cli("myquery", limit=5)

        expected_cmd = [
            "gh",
            "search",
            "repos",
            "myquery",
            "--json",
            "name,description,url,owner,stargazersCount,forksCount,visibility,isPrivate,updatedAt",
            "--limit",
            "5",
        ]
        mock_run.assert_called_once_with(
            expected_cmd, capture_output=True, text=True, check=True, env=unittest.mock.ANY
        )
        self.assertEqual(result, {"repositories": [{"name": "found_repo"}], "count": 1})
        passed_env = mock_run.call_args.kwargs["env"]
        self.assertEqual(passed_env.get("GITHUB_TOKEN"), "")

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    async def test_list_repos_using_gh_cli_error(self, mock_run):
        """Test handling errors from the gh list command."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh", stderr="gh error")

        with self.assertRaises(subprocess.CalledProcessError):
            await _list_repos_using_gh_cli()

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    async def test_search_repos_using_gh_cli_error(self, mock_run):
        """Test handling errors from the gh search command."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh", stderr="gh error")

        with self.assertRaises(subprocess.CalledProcessError):
            await _search_repos_using_gh_cli("query")

    @patch("src.cli_code.mcp.tools.examples.github.subprocess.run")
    async def test_list_repos_using_gh_cli_invalid_json(self, mock_run):
        """Test handling invalid JSON output from gh list."""
        mock_run.return_value = MockCompletedProcess(stdout="{invalid json", returncode=0)

        with self.assertRaises(json.JSONDecodeError):
            await _list_repos_using_gh_cli()


@pytest.mark.asyncio
class TestGitHubApiFunctions(unittest.TestCase):
    """Tests for functions interacting with GitHub API."""

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession")
    async def test_list_repos_using_api_authenticated(self, mock_session):
        mock_get = AsyncMock(return_value=MockAiohttpClientResponse(status=200, json_data=[{"name": "api_repo"}]))
        mock_session.return_value.__aenter__.return_value.get = mock_get

        result = await _list_repos_using_api()

        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        self.assertEqual(call_args[0], "https://api.github.com/user/repos")
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer test_token")
        self.assertEqual(result, {"repositories": [{"name": "api_repo"}], "count": 1})

    @patch.dict(os.environ, {}, clear=True)  # Ensure no token
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession")
    async def test_list_repos_using_api_for_user(self, mock_session):
        mock_get = AsyncMock(return_value=MockAiohttpClientResponse(status=200, json_data=[{"name": "user_api_repo"}]))
        mock_session.return_value.__aenter__.return_value.get = mock_get

        result = await _list_repos_using_api(username="testuser")

        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        self.assertEqual(call_args[0], "https://api.github.com/users/testuser/repos")
        self.assertNotIn("Authorization", call_kwargs["headers"])
        self.assertEqual(result, {"repositories": [{"name": "user_api_repo"}], "count": 1})

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession")
    async def test_list_repos_using_api_error(self, mock_session):
        mock_get = AsyncMock(return_value=MockAiohttpClientResponse(status=403, reason="Forbidden"))
        mock_session.return_value.__aenter__.return_value.get = mock_get

        with self.assertRaisesRegex(Exception, "GitHub API request failed: 403 Forbidden"):
            await _list_repos_using_api()

    @patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"})
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession")
    async def test_search_repos_using_api(self, mock_session):
        mock_get = AsyncMock(
            return_value=MockAiohttpClientResponse(
                status=200, json_data={"items": [{"name": "search_api_repo"}], "total_count": 1}
            )
        )
        mock_session.return_value.__aenter__.return_value.get = mock_get

        result = await _search_repos_using_api("myquery", limit=5)

        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        self.assertEqual(call_args[0], "https://api.github.com/search/repositories?q=myquery&per_page=5")
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer test_token")
        self.assertEqual(result, {"repositories": [{"name": "search_api_repo"}], "count": 1})

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession")
    async def test_search_repos_using_api_no_token(self, mock_session):
        mock_get = AsyncMock(
            return_value=MockAiohttpClientResponse(status=200, json_data={"items": [], "total_count": 0})
        )
        mock_session.return_value.__aenter__.return_value.get = mock_get

        await _search_repos_using_api("no_token_query")

        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        self.assertNotIn("Authorization", call_kwargs["headers"])

    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession")
    async def test_search_repos_using_api_error(self, mock_session):
        mock_get = AsyncMock(return_value=MockAiohttpClientResponse(status=401, reason="Unauthorized"))
        mock_session.return_value.__aenter__.return_value.get = mock_get

        with self.assertRaisesRegex(Exception, "GitHub API request failed: 401 Unauthorized"):
            await _search_repos_using_api("query")

    @patch.dict(os.environ, {}, clear=True)
    @patch("aiohttp.ClientSession.get")
    async def test_list_repos_using_api_connection_error(self, mock_get):
        """Test handling of connection errors during API list requests."""
        mock_get.side_effect = aiohttp.ClientConnectorError(None, OSError("Connection failed"))
        with self.assertRaisesRegex(Exception, "GitHub API request failed: Connection failed"):
            await _list_repos_using_api("user")

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.cli_code.mcp.tools.examples.github.aiohttp.ClientSession")
    async def test_list_repos_using_api_invalid_json(self, mock_session):
        """Test handling invalid JSON from API list response."""
        # Mock response that cannot be parsed as JSON
        mock_get = AsyncMock(return_value=MockAiohttpClientResponse(status=200, text_data="invalid json"))
        mock_session.return_value.__aenter__.return_value.get = mock_get

        with self.assertRaisesRegex(Exception, "GitHub API request failed: Failed to decode JSON"):
            await _list_repos_using_api("user")


@pytest.mark.asyncio
class TestGitHubHandlers(unittest.TestCase):
    """Tests for the main handler functions."""

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available", return_value=True)
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_gh_cli", new_callable=AsyncMock)
    async def test_list_repos_handler_uses_cli(self, mock_list_cli, mock_is_available):
        mock_list_cli.return_value = {"repositories": [], "count": 0}
        await github_list_repos_handler({"username": "user1"})
        mock_is_available.assert_called_once()
        mock_list_cli.assert_awaited_once_with("user1")

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available", return_value=False)
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_api", new_callable=AsyncMock)
    async def test_list_repos_handler_uses_api(self, mock_list_api, mock_is_available):
        mock_list_api.return_value = {"repositories": [], "count": 0}
        await github_list_repos_handler({"username": "user2"})
        mock_is_available.assert_called_once()
        mock_list_api.assert_awaited_once_with("user2")

    async def test_list_repos_handler_exception(self):
        with patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available", side_effect=Exception("Test Error")):
            with self.assertRaisesRegex(ValueError, "Failed to list GitHub repositories: Test Error"):
                await github_list_repos_handler({})

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available", return_value=True)
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_gh_cli", new_callable=AsyncMock)
    async def test_search_repos_handler_uses_cli(self, mock_search_cli, mock_is_available):
        mock_search_cli.return_value = {"repositories": [], "count": 0}
        await github_search_repos_handler({"query": "q1", "limit": 5})
        mock_is_available.assert_called_once()
        mock_search_cli.assert_awaited_once_with("q1", 5)

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available", return_value=False)
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_api", new_callable=AsyncMock)
    async def test_search_repos_handler_uses_api(self, mock_search_api, mock_is_available):
        mock_search_api.return_value = {"repositories": [], "count": 0}
        await github_search_repos_handler({"query": "q2", "limit": 15})
        mock_is_available.assert_called_once()
        mock_search_api.assert_awaited_once_with("q2", 15)

    async def test_search_repos_handler_no_query(self):
        with self.assertRaisesRegex(ValueError, "Query parameter is required"):
            await github_search_repos_handler({"limit": 10})

    async def test_search_repos_handler_exception(self):
        # Correctly structure nested contexts/assertions for exceptions
        with patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available", side_effect=Exception("Search Fail")):
            with self.assertRaisesRegex(ValueError, "Failed to search GitHub repositories: Search Fail"):
                await github_search_repos_handler({"query": "q"})


class TestGitHubToolClass(unittest.TestCase):
    """Tests for the GitHubTool class static methods."""

    def test_create_list_repos_tool(self):
        tool = GitHubTool.create_list_repos_tool()
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "github_list_repos")
        self.assertEqual(tool.handler, github_list_repos_handler)
        self.assertEqual(len(tool.parameters), 1)
        self.assertEqual(tool.parameters[0].name, "username")
        self.assertFalse(tool.parameters[0].required)

    def test_create_search_repos_tool(self):
        tool = GitHubTool.create_search_repos_tool()
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "github_search_repos")
        self.assertEqual(tool.handler, github_search_repos_handler)
        self.assertEqual(len(tool.parameters), 2)
        param_names = {p.name for p in tool.parameters}
        self.assertEqual(param_names, {"query", "limit"})
        query_param = next(p for p in tool.parameters if p.name == "query")
        limit_param = next(p for p in tool.parameters if p.name == "limit")
        self.assertTrue(query_param.required)
        self.assertFalse(limit_param.required)
        self.assertEqual(limit_param.type, "integer")


if __name__ == "__main__":
    unittest.main()
