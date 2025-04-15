"""
Constants for the models module.
"""

from enum import Enum, auto


class ToolResponseType(Enum):
    """Enum for types of tool responses."""
    SUCCESS = auto()
    ERROR = auto()
    USER_CONFIRMATION = auto()
    TASK_COMPLETE = auto() 