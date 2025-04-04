from abc import ABC, abstractmethod
from typing import Dict, List  # Add typing import

from rich.console import Console  # Import Console for type hinting


class AbstractModelAgent(ABC):
    """Abstract base class for different LLM provider agents."""

    def __init__(self, console: Console, model_name: str | None = None):
        """
        Initializes the agent.

        Args:
            console: The rich console object for output.
            model_name: The specific model ID to use (optional, uses provider default if None).
        """
        self.console = console
        self.model_name = model_name  # Store the specific model requested
        # History is now managed by subclasses
        # self.history = []

    @abstractmethod
    def generate(self, prompt: str) -> str | None:
        """
        Generate a response based on the user prompt and conversation history.
        This method should handle the agentic loop (API calls, tool calls).

        Args:
            prompt: The user's input prompt.

        Returns:
            The generated text response from the LLM, or None if an error occurs
            or the interaction doesn't result in a user-visible text response.
        """
        pass

    @abstractmethod
    def list_models(self) -> List[Dict] | None:  # Return list of dicts for more info
        """
        List available models for the provider.

        Returns:
            A list of dictionaries, each representing a model (e.g., {'id': 'model_id', 'name': 'Display Name'}),
            or None if listing fails.
        """
        pass

    # Removed shared history helper methods
    # def add_to_history(self, entry):
    #     ...
    # def clear_history(self):
    #    ...
