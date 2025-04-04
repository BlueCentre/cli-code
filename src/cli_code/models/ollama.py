import logging
import os
from rich.console import Console
from typing import List, Dict

# Attempt to import the OpenAI client library
try:
    import openai
    from openai import OpenAI # Import the client class
except ImportError:
    # This allows the module to be imported even if openai isn't installed yet,
    # but methods using it will fail later if it's not installed.
    openai = None
    OpenAI = None 
    # Log a warning or raise a more specific error during __init__ if openai is None.
    pass

from .base import AbstractModelAgent
from ..tools import AVAILABLE_TOOLS # Needed for tool schema generation

log = logging.getLogger(__name__)

class OllamaModel(AbstractModelAgent):
    """Interface for Ollama models using the OpenAI-compatible API."""

    def __init__(self, api_url: str, console: Console, model_name: str | None = None):
        """Initialize the Ollama model interface."""
        super().__init__(console=console, model_name=model_name) # Call base class init
        
        if not OpenAI:
             log.error("OpenAI client library not found. Please install it: pip install openai")
             raise ImportError("OpenAI client library is required for the Ollama provider. Please run: pip install openai")
             
        if not api_url:
            raise ValueError("Ollama API URL (base_url) is required.")
            
        self.api_url = api_url
        # self.model_name is set by super().__init__
        # The actual default model name from config should be resolved in main.py before passing
        
        try:
            # Initialize OpenAI client pointing to the Ollama base URL
            # Use a placeholder API key as Ollama doesn't require one by default
            self.client = OpenAI(
                base_url=self.api_url,
                api_key="ollama" # Required by the openai client, but value doesn't matter for Ollama
            )
            log.info(f"OpenAI client initialized for Ollama at: {self.api_url}")
            # Optionally, add a quick ping or model list check here to verify connection
            # self.list_models() # Could do a quick check
        except Exception as e:
            log.error(f"Failed to initialize OpenAI client for Ollama at {self.api_url}: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to Ollama API at {self.api_url}: {e}") from e

        # TODO: Add Ollama-specific tool/function calling setup if different from OpenAI standard
        # self.ollama_tools = self._prepare_ollama_tools()
        
        # TODO: Add Ollama-specific system prompt if needed
        # self.system_prompt = "..."
        # Add system prompt to history
        # self.add_to_history({"role": "system", "content": self.system_prompt})

        log.info(f"OllamaModel initialized for endpoint {self.api_url}")

    def generate(self, prompt: str) -> str | None:
        """
        Generate a response using the Ollama model.
        (Placeholder - requires implementation of API call, tool handling)
        """
        log.warning("OllamaModel.generate() is not yet implemented.")
        self.console.print("[bold yellow]Warning:[/bold yellow] Ollama provider text generation is not yet implemented.")
        # TODO: Implement the agentic loop for Ollama:
        # 1. Format history for OpenAI API
        # 2. Prepare tool schemas for OpenAI API
        # 3. Call client.chat.completions.create()
        # 4. Handle response (text or tool_calls)
        # 5. If tool_calls, execute tool, format result, add to history, call API again
        # 6. Return final text response
        return "(Ollama generate not implemented yet)"

    def list_models(self) -> List[Dict] | None:
        """
        List available models from the configured Ollama endpoint.
        """
        log.info(f"Attempting to list models from Ollama endpoint: {self.api_url}")
        if not self.client:
             log.error("OpenAI client not initialized for Ollama.")
             return None
        try:
            models_response = self.client.models.list()
            # The response object is a SyncPage[Model], access data via .data
            available_models = []
            for model in models_response.data:
                # Adapt the OpenAI Model object to our expected dict format
                model_info = {
                    "id": model.id,        # Typically the model identifier used in API calls
                    "name": getattr(model, 'name', model.id), # Use name if available, else id
                    # Add other potentially useful fields if needed, e.g., owner
                    # "owned_by": model.owned_by 
                }
                available_models.append(model_info)
            log.info(f"Found {len(available_models)} models at {self.api_url}")
            return available_models
        except Exception as e:
            log.error(f"Error listing models from Ollama at {self.api_url}: {e}", exc_info=True)
            self.console.print(f"[bold red]Error contacting Ollama endpoint '{self.api_url}':[/bold red] {e}")
            self.console.print("[yellow]Ensure the Ollama server is running and the API URL is correct.[/yellow]")
            return None # Indicate failure
    
    # TODO: Add helper methods for tool schema conversion, history formatting etc.
    # def _prepare_ollama_tools(self):
    #     ... 