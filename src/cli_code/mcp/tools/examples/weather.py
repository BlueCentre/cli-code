"""
Weather tool for MCP protocol.

This module provides a sample weather tool for demonstration.
"""

import json
from typing import Any, Dict

import aiohttp

from src.cli_code.mcp.tools.models import Tool, ToolParameter


async def weather_handler(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current weather information for a location.

    Args:
        parameters: Dictionary containing:
            location: The city or location to get weather for

    Returns:
        Weather data including temperature, conditions, and location info

    Raises:
        ValueError: If the location is invalid or connection fails
    """
    # Extract parameters
    location = parameters.get("location")

    # Validate required parameters
    if not location:
        raise ValueError("Location parameter is required")

    # In a real implementation, you would use an actual weather API
    # This is a mock implementation that returns fake data
    try:
        # Simulate API call with a small delay
        async with aiohttp.ClientSession() as session:
            # For demo purposes, we'll just simulate a response
            # In a real implementation, you would call a weather API like:
            # url = f"https://api.weather.com/v1/current?location={location}&apikey={API_KEY}"
            # async with session.get(url) as response:
            #     if response.status != 200:
            #         raise ValueError(f"Failed to get weather data: {response.status}")
            #     data = await response.json()

            # Simulated response
            weather_data = {
                "location": location,
                "current": {
                    "temperature": 72,
                    "temperature_unit": "F",
                    "conditions": "Partly Cloudy",
                    "humidity": 65,
                    "wind_speed": 5,
                    "wind_direction": "NE",
                },
                "forecast": [
                    {"day": "Today", "high": 75, "low": 60, "conditions": "Partly Cloudy"},
                    {"day": "Tomorrow", "high": 78, "low": 62, "conditions": "Sunny"},
                ],
            }

            return weather_data

    except Exception as e:
        raise ValueError(f"Failed to get weather data: {str(e)}") from e


class WeatherTool:
    """Weather tool for retrieving weather information."""

    @staticmethod
    def create() -> Tool:
        """
        Create a weather tool.

        Returns:
            A Tool instance for retrieving weather data
        """
        return Tool(
            name="weather",
            description="Gets current weather information for a specified location",
            parameters=[
                ToolParameter(
                    name="location",
                    description="City or location name to get weather for",
                    type="string",
                    required=True,
                )
            ],
            handler=weather_handler,
        )
