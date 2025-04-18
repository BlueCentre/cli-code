"""
Tests for the Weather example tool.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.cli_code.mcp.tools.examples.weather import WeatherTool, weather_handler
from src.cli_code.mcp.tools.models import Tool, ToolParameter


# Mock response class used in TestWeatherHandler
class MockAiohttpClientResponse:
    def __init__(self, status=200, json_data=None, text_data="", reason="OK"):
        self.status = status
        self._json_data = json_data
        self._text_data = text_data
        self.reason = reason

    async def json(self):
        if self._json_data is None:
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


@pytest.mark.asyncio
class TestWeatherHandler(unittest.TestCase):
    """Tests for the weather_handler async function."""

    @patch("src.cli_code.mcp.tools.examples.weather.aiohttp.ClientSession")
    async def test_weather_handler_success(self, mock_session):
        """Test successful weather retrieval (mocked)."""
        # Mock the session behavior (though currently unused by handler)
        mock_get = AsyncMock(return_value=MockAiohttpClientResponse(status=200, json_data={}))
        mock_session.return_value.__aenter__.return_value.get = mock_get

        params = {"location": "London"}
        result = await weather_handler(params)

        # Verify the mock response structure
        self.assertEqual(result["location"], "London")
        self.assertIn("current", result)
        self.assertIn("temperature", result["current"])
        self.assertIn("forecast", result)
        self.assertIsInstance(result["forecast"], list)
        # Check that the mock session wasn't actually called yet
        mock_get.assert_not_called()

    async def test_weather_handler_missing_location(self):
        """Test error when location parameter is missing."""
        params = {}
        with self.assertRaisesRegex(ValueError, "Location parameter is required"):
            await weather_handler(params)

    async def test_weather_handler_location_none(self):
        """Test error when location parameter is None."""
        params = {"location": None}
        with self.assertRaisesRegex(ValueError, "Location parameter is required"):
            await weather_handler(params)

    @patch(
        "src.cli_code.mcp.tools.examples.weather.aiohttp.ClientSession",
        side_effect=aiohttp.ClientError("Connection Error"),
    )
    async def test_weather_handler_api_exception(self, mock_session):
        """Test that exceptions during the (mocked) API call are caught."""
        # This tests the outer try/except block in the handler
        params = {"location": "London"}
        with self.assertRaisesRegex(ValueError, "Failed to get weather data: Connection Error"):
            await weather_handler(params)


class TestWeatherTool(unittest.TestCase):
    """Tests for the WeatherTool class."""

    def test_create(self):
        """Test the static create method."""
        tool = WeatherTool.create()

        # Check basic tool properties
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "weather")
        self.assertEqual(tool.description, "Gets current weather information for a specified location")
        self.assertEqual(tool.handler, weather_handler)

        # Check parameters
        self.assertEqual(len(tool.parameters), 1)

        location_param = tool.parameters[0]
        self.assertIsInstance(location_param, ToolParameter)
        self.assertEqual(location_param.name, "location")
        self.assertEqual(location_param.type, "string")
        self.assertTrue(location_param.required)

        # Check schema generation
        self.assertIsNotNone(tool.schema)
        self.assertEqual(tool.schema["name"], "weather")
        self.assertIn("parameters", tool.schema)
        schema_params = tool.schema["parameters"]
        self.assertEqual(schema_params["type"], "object")
        self.assertCountEqual(schema_params["required"], ["location"])
        self.assertIn("location", schema_params["properties"])
        self.assertEqual(schema_params["properties"]["location"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
