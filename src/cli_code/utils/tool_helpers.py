"""
Utility functions for working with tools.

This module provides helper functions for working with the CLI Code tool system.
"""

import logging
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


def execute_tool(tool_name: str, args: Dict[str, Any] = None) -> Any:
    """
    Execute a tool with the given arguments.

    This is a helper function to execute a tool by name with the given arguments.
    It's used in tests and possibly elsewhere for simplified tool execution.

    Args:
        tool_name: The name of the tool to execute
        args: Optional dictionary of arguments to pass to the tool

    Returns:
        Any: The result of the tool execution

    Raises:
        ValueError: If the tool does not exist
    """
    from ..tools import get_tool

    tool = get_tool(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")

    return tool.execute(**(args or {}))
