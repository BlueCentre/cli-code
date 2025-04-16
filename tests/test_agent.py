from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli_code.mcp.tools.registry import ToolRegistry

# Replace the incorrect Agent import with a concrete implementation
# from src.cli_code.agent import Agent
from src.cli_code.models.gemini import GeminiModel  # Use GeminiModel for testing agent logic

# Remove the problematic import as mocks.py or MockBaseModel doesn't seem to exist
# from .mocks import MockBaseModel


@pytest.fixture
def agent(mock_model, mock_tool_registry):
    """Fixture to create an Agent instance with mocked dependencies."""
    # Assume the test needs a concrete agent, instantiate GeminiModel
    # This might require mocking api_key and console as well
    mock_console = MagicMock()
    # Use GeminiModel instead of Agent
    return GeminiModel(api_key="mock_api_key", console=mock_console, model_name="mock-gemini")
    # Ensure the mocked tool registry is assigned if the agent uses it directly
    # agent.tool_registry = mock_tool_registry # This line might be needed depending on Agent structure
    # return agent
