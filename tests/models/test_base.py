"""
Tests for the AbstractModelAgent base class.
"""

import pytest
from rich.console import Console

from src.cli_code.models.base import AbstractModelAgent


class ConcreteModelAgent(AbstractModelAgent):
    """Concrete implementation of AbstractModelAgent for testing."""

    def __init__(self, console, model_name=None):
        super().__init__(console, model_name)
        # Initialize any specific attributes for testing
        self.history = []

    def generate(self, prompt: str) -> str | None:
        """Implementation of abstract method."""
        return f"Generated response for: {prompt}"

    def list_models(self):
        """Implementation of abstract method."""
        return [{"id": "model1", "name": "Test Model 1"}, {"id": "model2", "name": "Test Model 2"}]


@pytest.fixture
def mock_console(mocker):
    """Provides a mocked Console object."""
    return mocker.MagicMock(spec=Console)


@pytest.fixture
def model_agent(mock_console):
    """Provides a concrete model agent instance for testing."""
    return ConcreteModelAgent(mock_console, "test-model")


def test_initialization(mock_console):
    """Test initialization of the AbstractModelAgent."""
    model = ConcreteModelAgent(mock_console, "test-model")

    # Check initialized attributes
    assert model.console == mock_console
    assert model.model_name == "test-model"


def test_generate_method(model_agent):
    """Test the concrete implementation of the generate method."""
    response = model_agent.generate("Test prompt")
    assert response == "Generated response for: Test prompt"


def test_list_models_method(model_agent):
    """Test the concrete implementation of the list_models method."""
    models = model_agent.list_models()

    # Verify structure and content
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0]["id"] == "model1"
    assert models[1]["name"] == "Test Model 2"
