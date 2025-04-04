from abc import ABC, abstractmethod
from rich.console import Console # Import Console for type hinting

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
        self.model_name = model_name # Store the specific model requested
        self.history = [] # Initialize chat history
        # Provider-specific initialization (e.g., API client) should happen in subclass __init__

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
    def list_models(self) -> list[dict] | None: # Return list of dicts for more info
        """
        List available models for the provider.

        Returns:
            A list of dictionaries, each representing a model (e.g., {'id': 'model_id', 'name': 'Display Name'}),
            or None if listing fails.
        """
        pass

    # Helper methods common to agents could be added here (e.g., history management)
    def add_to_history(self, entry):
        """Adds an entry to the conversation history."""
        # Basic history management - could be expanded (e.g., token limits)
        self.history.append(entry)
        # Simple truncation for now - TODO: Implement token-based truncation
        MAX_HISTORY = 20
        if len(self.history) > MAX_HISTORY:
            # Keep system prompt (if any) and N most recent turns
            # Assuming system prompt might be history[0] - adjust if needed
            # A more robust implementation would identify the system prompt role
            self.history = [h for i, h in enumerate(self.history) if i == 0 or i > len(self.history) - (MAX_HISTORY -1) ]

    def clear_history(self):
        """Clears the conversation history."""
        self.history = [] 