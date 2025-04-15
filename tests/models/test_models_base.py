"""
Tests for the AbstractModelAgent base class.
"""

from unittest.mock import MagicMock

import pytest

# Direct import for coverage tracking
import src.cli_code.models.base
from src.cli_code.models.base import AbstractModelAgent


class TestModelImplementation(AbstractModelAgent):
    """A concrete implementation of AbstractModelAgent for testing."""

    def generate(self, prompt):
        """Test implementation of the generate method."""
        return f"Response to: {prompt}"

    def list_models(self):
        """Test implementation of the list_models method."""
        return [{"name": "test-model", "displayName": "Test Model"}]


def test_abstract_model_init():
    """Test initialization of a concrete model implementation."""
    console = MagicMock()
    model = TestModelImplementation(console=console, model_name="test-model")

    assert model.console == console
    assert model.model_name == "test-model"


def test_generate_method():
    """Test the generate method of the concrete implementation."""
    model = TestModelImplementation(console=MagicMock(), model_name="test-model")
    response = model.generate("Hello")

    assert response == "Response to: Hello"


def test_list_models_method():
    """Test the list_models method of the concrete implementation."""
    model = TestModelImplementation(console=MagicMock(), model_name="test-model")
    models = model.list_models()

    assert len(models) == 1
    assert models[0]["name"] == "test-model"
    assert models[0]["displayName"] == "Test Model"


def test_abstract_class_methods():
    """Test that AbstractModelAgent cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AbstractModelAgent(console=MagicMock(), model_name="test-model")
