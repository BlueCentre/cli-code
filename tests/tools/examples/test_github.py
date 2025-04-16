"""
Tests for the GitHub tool example.
"""

import asyncio
import json
import os
import subprocess
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# Modules to test
from src.cli_code.mcp.tools.examples import github


# Helper to create mock subprocess result
def create_subprocess_result(stdout="", stderr="", returncode=0):
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


@pytest.mark.usefixtures("mock_subprocess_run")  # Use fixture for cleaner mocking
class TestGitHubCliFunctions:
    """Tests for functions interacting with the gh CLI."""

    @pytest.fixture(autouse=True)
    def mock_subprocess_run(self, monkeypatch):
        """Fixture to mock subprocess.run."""
        mock_run = MagicMock(spec=subprocess.run)
        monkeypatch.setattr(subprocess, "run", mock_run)
        # Default successful run
        mock_run.return_value = create_subprocess_result(stdout="[]", returncode=0)
        yield mock_run  # Provide the mock for assertions

    def test_is_gh_cli_available_true(self, mock_subprocess_run):
        """Test _is_gh_cli_available when gh CLI is present."""
        mock_subprocess_run.return_value = create_subprocess_result(returncode=0)
        assert github._is_gh_cli_available() is True
        mock_subprocess_run.assert_called_once_with(["gh", "--version"], capture_output=True, text=True, check=False)

    def test_is_gh_cli_available_false(self, mock_subprocess_run):
        """Test _is_gh_cli_available when gh CLI is not present (non-zero return)."""
        mock_subprocess_run.return_value = create_subprocess_result(returncode=1)
        assert github._is_gh_cli_available() is False
        mock_subprocess_run.assert_called_once_with(["gh", "--version"], capture_output=True, text=True, check=False)

    def test_is_gh_cli_available_file_not_found(self, mock_subprocess_run):
        """Test _is_gh_cli_available when gh command raises FileNotFoundError."""
        mock_subprocess_run.side_effect = FileNotFoundError
        assert github._is_gh_cli_available() is False
        mock_subprocess_run.assert_called_once_with(["gh", "--version"], capture_output=True, text=True, check=False)

    # --- Tests for _list_repos_using_gh_cli ---
    @pytest.mark.asyncio
    async def test_list_repos_using_gh_cli_no_user(self, mock_subprocess_run):
        """Test listing repos for authenticated user via CLI."""
        expected_repos = [{"name": "repo1", "description": "Desc 1"}]
        mock_subprocess_run.return_value = create_subprocess_result(stdout=json.dumps(expected_repos), returncode=0)

        result = await github._list_repos_using_gh_cli()

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        # Check base command and JSON format flag
        assert call_args[:4] == ["gh", "repo", "list", "--json"]
        # Ensure username is NOT present by checking the element after 'list'
        # The expected command when no user is provided:
        # ['gh', 'repo', 'list', '--json', 'name,...', '--limit', '100']
        assert call_args[3] == "--json"  # Check that the user position is occupied by --json
        assert "--limit" in call_args
        assert str(github.DEFAULT_GH_LIST_LIMIT) in call_args
        # Check environment for unset GITHUB_TOKEN
        env = mock_subprocess_run.call_args[1].get("env")
        assert env is not None
        assert env.get("GITHUB_TOKEN") == ""

        assert result == {"repositories": expected_repos, "count": len(expected_repos)}

    @pytest.mark.asyncio
    async def test_list_repos_using_gh_cli_with_user(self, mock_subprocess_run):
        """Test listing repos for a specific user via CLI."""
        test_user = "octocat"
        expected_repos = [{"name": "octo-repo", "description": "Octo Desc"}]
        mock_subprocess_run.return_value = create_subprocess_result(stdout=json.dumps(expected_repos), returncode=0)

        result = await github._list_repos_using_gh_cli(username=test_user, limit=50)

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        # Check base command, user, and JSON format flag
        assert call_args[:5] == ["gh", "repo", "list", test_user, "--json"]
        assert "--limit" in call_args
        assert "50" in call_args  # Check specific limit passed
        # Check environment
        env = mock_subprocess_run.call_args[1].get("env")
        assert env is not None
        assert env.get("GITHUB_TOKEN") == ""

        assert result == {"repositories": expected_repos, "count": len(expected_repos)}

    @pytest.mark.asyncio
    async def test_list_repos_using_gh_cli_error(self, mock_subprocess_run):
        """Test handling non-zero return code from gh CLI list command."""
        mock_subprocess_run.return_value = create_subprocess_result(stderr="gh command failed", returncode=1)
        # Make check=True raise the error
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["gh", "repo", "list"], stderr="gh command failed"
        )

        with pytest.raises(subprocess.CalledProcessError):
            await github._list_repos_using_gh_cli()

    @pytest.mark.asyncio
    async def test_list_repos_using_gh_cli_invalid_json(self, mock_subprocess_run):
        """Test handling invalid JSON output from gh CLI list command."""
        mock_subprocess_run.return_value = create_subprocess_result(stdout="this is not json", returncode=0)

        with pytest.raises(json.JSONDecodeError):
            await github._list_repos_using_gh_cli()

    # --- Tests for _search_repos_using_gh_cli ---
    @pytest.mark.asyncio
    async def test_search_repos_using_gh_cli_success(self, mock_subprocess_run):
        """Test successful repo search via CLI."""
        query = "cli-code"
        expected_repos = [{"fullName": "james/cli-code", "description": "Test repo"}]
        mock_subprocess_run.return_value = create_subprocess_result(
            stdout=json.dumps(expected_repos),
            returncode=0,  # API returns items directly now
        )

        result = await github._search_repos_using_gh_cli(query)

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        # Check the full command including the default limit
        assert call_args == [
            "gh",
            "search",
            "repos",
            query,
            "--json",
            "fullName,description",
            "--limit",
            str(github.DEFAULT_GH_SEARCH_LIMIT),
        ]
        # Check environment
        env = mock_subprocess_run.call_args[1].get("env")
        assert env is not None
        assert env.get("GITHUB_TOKEN") == ""

        # Update expected result structure to match function's return
        expected_result = {"query": query, "results": expected_repos, "count": len(expected_repos)}
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_search_repos_using_gh_cli_error(self, mock_subprocess_run):
        """Test handling non-zero return code from gh CLI search command."""
        query = "cli-code"
        mock_subprocess_run.return_value = create_subprocess_result(stderr="gh search failed", returncode=1)
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["gh", "search", "repos"], stderr="gh search failed"
        )

        with pytest.raises(subprocess.CalledProcessError):
            await github._search_repos_using_gh_cli(query)

    @pytest.mark.asyncio
    async def test_search_repos_using_gh_cli_invalid_json(self, mock_subprocess_run):
        """Test handling invalid JSON output from gh CLI search command."""
        query = "cli-code"
        mock_subprocess_run.return_value = create_subprocess_result(stdout="not valid json", returncode=0)

        with pytest.raises(json.JSONDecodeError):
            await github._search_repos_using_gh_cli(query)


# Async tests need pytest-asyncio
@pytest.fixture
def mock_aiohttp_session_github():
    """Fixture to mock aiohttp.ClientSession specifically for GitHub API tests."""
    mock_post_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_post_response.status = 200  # Default success
    mock_post_response.json = AsyncMock(return_value={})
    mock_post_response.text = AsyncMock(return_value="{}")
    mock_post_response.headers = {}
    mock_post_response.reason = "OK"  # Default reason

    mock_get_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_get_response.status = 200  # Default success
    mock_get_response.json = AsyncMock(return_value=[])  # Default: empty list
    mock_get_response.text = AsyncMock(return_value="[]")
    mock_get_response.headers = {}
    mock_get_response.reason = "OK"  # Default reason

    # --- Configure raise_for_status --- START
    def raise_for_status_side_effect(response):
        # Helper to simulate raise_for_status based on the response's status code
        if response.status >= 400:
            # Try to get a message from the mocked json response's configured return_value
            message = None
            try:
                # Access configured return value directly from the AsyncMock
                error_payload = response.json.return_value
                if isinstance(error_payload, dict):
                    message = error_payload.get("message")
            except Exception as e:
                # Log if accessing return_value unexpectedly fails
                print(f"Error accessing mock json return value: {e}")
                pass  # Fall through to use reason

            # Use the extracted message or fall back to reason
            final_message = message or response.reason or f"HTTP Error {response.status}"

            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),  # Mock request_info
                history=(),
                status=response.status,
                message=final_message,
                headers=response.headers,
            )

    mock_post_response.raise_for_status = MagicMock(
        side_effect=lambda: raise_for_status_side_effect(mock_post_response)
    )
    mock_get_response.raise_for_status = MagicMock(side_effect=lambda: raise_for_status_side_effect(mock_get_response))
    # --- Configure raise_for_status --- END

    # Mock the async context manager for post/get responses
    mock_post_response_cm = AsyncMock()
    mock_post_response_cm.__aenter__.return_value = mock_post_response
    mock_post_response_cm.__aexit__.return_value = None

    mock_get_response_cm = AsyncMock()
    mock_get_response_cm.__aenter__.return_value = mock_get_response
    mock_get_response_cm.__aexit__.return_value = None

    # Mock the ClientSession itself
    mock_session = MagicMock(spec=aiohttp.ClientSession)
    mock_session.post.return_value = mock_post_response_cm
    mock_session.get.return_value = mock_get_response_cm

    # Mock the async context manager for the ClientSession
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_session_cm.__aexit__.return_value = None

    with patch("aiohttp.ClientSession", return_value=mock_session_cm) as mock_client_session_cls:
        # Yield the session mock and the specific response mocks
        yield mock_session, mock_post_response, mock_get_response


@pytest.mark.usefixtures("mock_aiohttp_session_github")
class TestGitHubApiFunctions:
    """Tests for functions interacting with the GitHub API via aiohttp."""

    # --- TODO: Add tests for _list_repos_using_api ---
    @pytest.mark.asyncio
    async def test_list_repos_using_api_authenticated(self, mock_aiohttp_session_github, monkeypatch):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_list_repos_using_api_for_user(self, mock_aiohttp_session_github):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_list_repos_using_api_error(self, mock_aiohttp_session_github):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_list_repos_using_api_connection_error(self, mock_aiohttp_session_github):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_list_repos_using_api_invalid_json(self, mock_aiohttp_session_github):
        pass  # Placeholder

    # --- TODO: Add tests for _search_repos_using_api ---
    @pytest.mark.asyncio
    async def test_search_repos_using_api(self, mock_aiohttp_session_github, monkeypatch):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_search_repos_using_api_no_token(self, mock_aiohttp_session_github, monkeypatch):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_search_repos_using_api_error(self, mock_aiohttp_session_github):
        pass  # Placeholder


class TestGitHubHandlers:
    """Tests for the main handler dispatch functions."""

    # --- TODO: Add tests for github_list_repos_handler ---
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_gh_cli")
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_api")
    @pytest.mark.asyncio
    async def test_list_repos_handler_uses_cli(self, mock_api, mock_cli, mock_is_avail):
        pass  # Placeholder

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_gh_cli")
    @patch("src.cli_code.mcp.tools.examples.github._list_repos_using_api")
    @pytest.mark.asyncio
    async def test_list_repos_handler_uses_api(self, mock_api, mock_cli, mock_is_avail):
        pass  # Placeholder

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @pytest.mark.asyncio
    async def test_list_repos_handler_exception(self, mock_is_avail):
        pass  # Placeholder

    # --- TODO: Add tests for github_search_repos_handler ---
    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_gh_cli")
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_api")
    @pytest.mark.asyncio
    async def test_search_repos_handler_uses_cli(self, mock_api, mock_cli, mock_is_avail):
        pass  # Placeholder

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_gh_cli")
    @patch("src.cli_code.mcp.tools.examples.github._search_repos_using_api")
    @pytest.mark.asyncio
    async def test_search_repos_handler_uses_api(self, mock_api, mock_cli, mock_is_avail):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_search_repos_handler_no_query(self):
        pass  # Placeholder

    @patch("src.cli_code.mcp.tools.examples.github._is_gh_cli_available")
    @pytest.mark.asyncio
    async def test_search_repos_handler_exception(self, mock_is_avail):
        pass  # Placeholder


# --- TODO: Add tests for GitHubTool class ---
class TestGitHubToolClass:
    """Tests for the GitHubTool class itself."""

    def test_create_list_repos_tool(self):
        pass  # Placeholder

    def test_create_search_repos_tool(self):
        pass  # Placeholder

    @patch("src.cli_code.mcp.tools.examples.github.github_list_repos_handler")
    @patch("src.cli_code.mcp.tools.examples.github.github_search_repos_handler")
    @pytest.mark.asyncio
    async def test_execute_list(self, mock_search, mock_list):
        pass  # Placeholder

    @patch("src.cli_code.mcp.tools.examples.github.github_list_repos_handler")
    @patch("src.cli_code.mcp.tools.examples.github.github_search_repos_handler")
    @pytest.mark.asyncio
    async def test_execute_search(self, mock_search, mock_list):
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_execute_invalid_op(self):
        pass  # Placeholder


# --- Tests for GitHub REST API Functions ---


class TestGitHubRestFunctions:
    """Tests for functions interacting with the GitHub REST API."""

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._get_github_token", return_value="fake-token")
    async def test_create_issue_success(self, mock_get_token, mock_aiohttp_session_github):
        """Test successful issue creation via REST API."""
        owner = "test-owner"
        repo = "test-repo"
        title = "Test Issue Title"
        body = "Test issue body."
        labels = ["bug", "testing"]
        assignees = ["test-user"]
        expected_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        expected_data = {"title": title, "body": body, "labels": labels, "assignees": assignees}
        mock_response = {"html_url": f"https://github.com/{owner}/{repo}/issues/1", "number": 1}

        # Configure the mock session
        mock_session, mock_post_resp, _ = mock_aiohttp_session_github  # Unpack fixture correctly
        mock_post_resp.status = 201
        mock_post_resp.json = AsyncMock(return_value=mock_response)  # Configure json mock

        result = await github._create_issue_using_rest(
            owner=owner, repo=repo, title=title, body=body, labels=labels, assignees=assignees
        )

        expected_headers = github.GITHUB_API_HEADERS.copy()
        expected_headers["Authorization"] = "Bearer fake-token"
        mock_session.post.assert_called_once_with(
            expected_url,
            json=expected_data,
            headers=expected_headers,  # Expect Authorization header
        )
        assert result == mock_response

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._get_github_token", return_value="fake-token")
    async def test_create_issue_api_error(self, mock_get_token, mock_aiohttp_session_github):
        """Test handling API error during issue creation."""
        owner = "test-owner"
        repo = "test-repo"
        title = "Test Issue Title"
        error_message = "API rate limit exceeded"

        # Configure the mock session for an error
        mock_session, mock_post_resp, _ = mock_aiohttp_session_github  # Unpack fixture correctly
        mock_post_resp.status = 403
        mock_post_resp.reason = "Forbidden"
        mock_post_resp.json = AsyncMock(return_value={"message": error_message})  # Mock json response for error

        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await github._create_issue_using_rest(owner=owner, repo=repo, title=title)

        assert excinfo.value.status == 403
        assert excinfo.value.message == error_message
        assert excinfo.value.headers == {}  # Check for empty dict, not None

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._get_github_token", return_value="fake-token")
    async def test_search_issues_success(self, mock_get_token, mock_aiohttp_session_github):
        """Test successful issue search via REST API."""
        query = "label:bug state:open repo:test-owner/test-repo"
        sort = "created"
        order = "desc"
        expected_url = f"https://api.github.com/search/issues"
        expected_params = {"q": query, "sort": sort, "order": order, "per_page": github.DEFAULT_GH_SEARCH_LIMIT}
        mock_response = {
            "total_count": 1,
            "incomplete_results": False,
            "items": [{"html_url": f"https://github.com/test-owner/test-repo/issues/1", "number": 1}],
        }

        mock_session, _, mock_get_resp = mock_aiohttp_session_github  # Unpack fixture correctly
        mock_get_resp.status = 200
        mock_get_resp.json = AsyncMock(return_value=mock_response)  # Configure json mock

        result = await github._search_issues_using_rest(query=query, sort=sort, order=order)

        expected_headers = github.GITHUB_API_HEADERS.copy()
        expected_headers["Authorization"] = "Bearer fake-token"
        mock_session.get.assert_called_once_with(
            expected_url,
            params=expected_params,
            headers=expected_headers,  # Expect Authorization header
        )
        assert result == mock_response

    @pytest.mark.asyncio
    @patch("src.cli_code.mcp.tools.examples.github._get_github_token", return_value="fake-token")
    async def test_search_issues_api_error(self, mock_get_token, mock_aiohttp_session_github):
        """Test handling API error during issue search."""
        query = "invalid-query"
        error_message = "Validation Failed"

        mock_session, _, mock_get_resp = mock_aiohttp_session_github  # Unpack fixture correctly
        mock_get_resp.status = 422
        mock_get_resp.reason = "Unprocessable Entity"
        mock_get_resp.json = AsyncMock(return_value={"message": error_message})  # Mock json response for error

        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await github._search_issues_using_rest(query=query)

        assert excinfo.value.status == 422
        assert excinfo.value.message == error_message
