"""
Tests for the Weather Tool.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiohttp
from src.cli_code.mcp.tools.examples.weather import WeatherTool, weather_handler
from src.cli_code.mcp.tools.models import Tool


class TestWeatherHandler:
    @pytest.mark.asyncio
    async def test_get_weather_success(self):
        """Test successful weather retrieval."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "location": {
                "name": "New York",
                "region": "New York",
                "country": "United States of America"
            },
            "current": {
                "temp_c": 25.0,
                "temp_f": 77.0,
                "condition": {
                    "text": "Sunny",
                    "icon": "//cdn.weatherapi.com/weather/64x64/day/113.png"
                },
                "humidity": 60,
                "feelslike_c": 26.0,
                "feelslike_f": 78.8
            }
        })
        
        # Mock the aiohttp.ClientSession
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=mock_response)
        
        # Call the handler
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await weather_handler({"location": "New York"})
        
        # Verify the result
        assert result["location"] == "New York, New York, United States of America"
        assert result["temperature"]["celsius"] == 25.0
        assert result["temperature"]["fahrenheit"] == 77.0
        assert result["condition"] == "Sunny"
        assert result["humidity"] == 60
        assert result["feels_like"]["celsius"] == 26.0
        assert result["feels_like"]["fahrenheit"] == 78.8
    
    @pytest.mark.asyncio
    async def test_missing_location(self):
        """Test missing location parameter."""
        with pytest.raises(ValueError, match="Missing required parameter: location"):
            await weather_handler({})
    
    @pytest.mark.asyncio
    async def test_api_error(self):
        """Test handling of weather API errors."""
        # Create mock response for error
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"error": {"message": "Invalid location"}})
        
        # Mock the aiohttp.ClientSession
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=mock_response)
        
        # Call the handler
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(Exception, match="Weather API error: Invalid location"):
                await weather_handler({"location": "InvalidLocation"})
    
    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test handling of connection errors."""
        # Mock the aiohttp.ClientSession to raise an exception
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(side_effect=aiohttp.ClientError("Connection error"))
        
        # Call the handler
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(Exception, match="Failed to connect to weather API"):
                await weather_handler({"location": "New York"})


class TestWeatherTool(unittest.TestCase):
    """Tests for the WeatherTool class."""

    def test_create(self):
        """Test creation of the weather tool."""
        tool = WeatherTool.create()
        
        # Check tool properties
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "weather")
        self.assertIn("weather information", tool.description.lower())
        
        # Check parameters
        self.assertEqual(len(tool.parameters), 1)
        param = tool.parameters[0]
        self.assertEqual(param.name, "location")
        self.assertTrue(param.required)
        
        # Check handler setup
        self.assertEqual(tool.handler.__name__, "weather_handler")

    @pytest.mark.asyncio
    @patch('src.cli_code.mcp.tools.examples.weather.weather_handler')
    async def test_tool_execution(self, mock_handler):
        """Test executing the weather tool."""
        # Setup mock
        expected_result = {
            "location": "San Francisco",
            "current": {"temperature": 65},
            "forecast": [{"day": "Today", "high": 68, "low": 55}]
        }
        mock_handler.return_value = expected_result
        
        # Create tool and execute
        tool = WeatherTool.create()
        result = await tool.execute({"location": "San Francisco"})
        
        # Verify result
        self.assertEqual(result, expected_result)
        mock_handler.assert_called_once_with({"location": "San Francisco"}) 