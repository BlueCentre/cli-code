import logging
import os
from rich.console import Console
from typing import List, Dict
import json # For formatting tool results

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
from ..tools import AVAILABLE_TOOLS, get_tool # Import get_tool

log = logging.getLogger(__name__)
MAX_OLLAMA_ITERATIONS = 5 # Limit tool call loops for Ollama initially

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

        # --- Initialize Ollama-specific History (OpenAI format) ---
        self.history = []
        # Add system prompt if needed (using "system" role)
        # TODO: Create a good default system prompt for Ollama + Tools
        # self.system_prompt = "You are a helpful assistant that can use tools."
        # self.add_to_history({"role": "system", "content": self.system_prompt})
        log.info(f"OllamaModel initialized for endpoint {self.api_url}")

    def generate(self, prompt: str) -> str | None:
        """Generate a response using the Ollama model via OpenAI API format."""
        if not self.client:
            log.error("Ollama generate called but OpenAI client not initialized.")
            return "Error: Ollama client not initialized."
            
        # Ensure model name is set (either from constructor or config default resolved in main)
        if not self.model_name:
             log.error("Ollama generate called without a model name specified.")
             # Try getting default from config as a fallback, though main.py should handle this
             # config = Config() # Need config access or pass it in
             # self.model_name = config.get_default_model(provider='ollama') 
             # if not self.model_name:
             return "Error: No Ollama model name configured or specified."

        log.info(f"Ollama Agent Loop - Processing prompt: '{prompt[:100]}...' using model '{self.model_name}'")

        # === Step 1: Mandatory Orientation (if desired for Ollama) ===
        # Decide if Ollama should also perform an initial 'ls' like Gemini
        # For simplicity, let's skip it for now and add the user prompt directly.
        # orientation_context = ""
        # try:
        #     ls_tool = get_tool("ls")
        #     if ls_tool: ls_result = ls_tool.execute(); orientation_context = f"Current directory:\n{ls_result}\n"
        # except Exception as e: log.error(f"Ollama initial ls failed: {e}")
        # full_prompt = f"{orientation_context}\nUser request: {prompt}"
        # self.add_to_history({"role": "user", "content": full_prompt})

        # Add the user prompt directly to history
        self.add_to_history({"role": "user", "content": prompt})
        
        iteration_count = 0
        while iteration_count < MAX_OLLAMA_ITERATIONS:
             iteration_count += 1
             log.info(f"Ollama Agent Iteration {iteration_count}/{MAX_OLLAMA_ITERATIONS}")

             try:
                 # === Prepare Tools for API Call ===
                 # Convert our tool schemas to OpenAI format
                 current_tools = self._prepare_openai_tools()

                 # === Call Ollama (OpenAI API) ===
                 log.debug(f"Sending request to Ollama. Model: {self.model_name}. History: {self.history}. Tools: {current_tools}")
                 with self.console.status(f"[yellow]Ollama thinking ({self.model_name})...", spinner="dots"):
                     response = self.client.chat.completions.create(
                         model=self.model_name,
                         messages=self.history,
                         tools=current_tools,
                         tool_choice="auto" # Let the model decide if it needs tools
                         # Add other parameters like temperature if needed
                     )
                 log.debug(f"Raw Ollama Response (Iter {iteration_count}): {response}")

                 response_message = response.choices[0].message
                 
                 # === Handle Response === 
                 tool_calls = response_message.tool_calls
                 if tool_calls:
                     # === Tool Call Requested ===
                     log.info(f"Ollama requested {len(tool_calls)} tool call(s).")
                     # Add the assistant's response (containing the tool requests) to history
                     self.add_to_history(response_message.model_dump(exclude_unset=True)) 

                     # --- Execute Tools --- 
                     for tool_call in tool_calls:
                         tool_name = tool_call.function.name
                         tool_args_str = tool_call.function.arguments
                         tool_call_id = tool_call.id
                         log.info(f"Processing tool call: ID={tool_call_id}, Name={tool_name}, Args='{tool_args_str}'")

                         try:
                             tool_args = json.loads(tool_args_str)
                         except json.JSONDecodeError:
                             log.error(f"Failed to decode JSON arguments for tool {tool_name}: {tool_args_str}")
                             tool_result = f"Error: Invalid JSON arguments provided: {tool_args_str}"
                             tool_error = True
                         else:
                             # --- Execute the actual tool --- 
                             tool_instance = get_tool(tool_name)
                             if tool_instance:
                                  try:
                                       # TODO: Add Human-in-the-Loop confirmation here if needed for Ollama
                                       # if tool_name in ["edit", "create_file"]: ... confirmation logic ...
                                       
                                       # === ADD STATUS FOR TOOL EXEC ===
                                       with self.console.status(f"[cyan]Executing tool: {tool_name}...", spinner="dots"):
                                            tool_result = tool_instance.execute(**tool_args)
                                       # === END STATUS ===
                                       log.info(f"Tool {tool_name} executed successfully. Result length: {len(tool_result) if tool_result else 0}")
                                       tool_error = False 
                                  except Exception as tool_exec_error:
                                       log.error(f"Error executing tool {tool_name} with args {tool_args}: {tool_exec_error}", exc_info=True)
                                       tool_result = f"Error executing tool {tool_name}: {str(tool_exec_error)}"
                                       tool_error = True
                             else:
                                  log.error(f"Tool '{tool_name}' requested by Ollama not found in available tools.")
                                  tool_result = f"Error: Tool '{tool_name}' not found."
                                  tool_error = True

                         # --- Add Tool Result to History --- 
                         self.add_to_history(
                             {
                                 "tool_call_id": tool_call_id,
                                 "role": "tool",
                                 "name": tool_name,
                                 "content": tool_result, # Send back the execution result
                             }
                         )
                     # --- Loop back to LLM --- 
                     continue # Continue the while loop to send tool results back to Ollama

                 else:
                     # === Text Response Received ===
                     final_text = response_message.content
                     log.info(f"Ollama returned final text response: {final_text[:100]}...")
                     # Add assistant's final text response to history
                     self.add_to_history(response_message.model_dump(exclude_unset=True)) 
                     return final_text # Exit loop and return text

             except Exception as e:
                 log.error(f"Error during Ollama agent iteration {iteration_count}: {e}", exc_info=True)
                 self.console.print(f"[bold red]Error during Ollama interaction:[/bold red] {e}")
                 # Clean history? Pop last user message?
                 if self.history and self.history[-1].get("role") == "user":
                      self.history.pop()
                 return f"(Error interacting with Ollama: {e})" # Return error message
        
        # If loop finishes without returning text (e.g., max iterations)
        log.warning(f"Ollama agent loop reached max iterations ({MAX_OLLAMA_ITERATIONS}).")
        return "(Agent reached maximum iterations)"

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

    # --- Ollama-specific history management ---
    def add_to_history(self, message: Dict):
        """Adds a message dictionary (OpenAI format) to the history."""
        # Basic validation
        if not isinstance(message, dict) or "role" not in message:
             log.warning(f"Attempted to add invalid message to Ollama history: {message}")
             return
        self.history.append(message)
        # TODO: Implement context window management for Ollama (token counting)
        # self._manage_ollama_context()

    def clear_history(self):
        """Clears the Ollama conversation history."""
        self.history = []
        # Re-add system prompt if we are using one
        # if hasattr(self, 'system_prompt') and self.system_prompt:
        #     self.add_to_history({"role": "system", "content": self.system_prompt})
        log.info("Ollama history cleared.")

    # --- Tool Preparation Helper ---
    def _prepare_openai_tools(self) -> List[Dict] | None:
        """Converts available tools to OpenAI tool format."""
        if not AVAILABLE_TOOLS:
            return None
        
        openai_tools = []
        for name, tool_instance in AVAILABLE_TOOLS.items():
            try:
                # Assuming get_function_declaration returns something convertible
                # Needs adjustment based on actual FunctionDeclaration structure
                declaration = tool_instance.get_function_declaration()
                if declaration:
                    # Convert Gemini FunctionDeclaration to OpenAI tool format
                    # This is a potential point of complexity/error
                    # Example assumes declaration has .name, .description, .parameters (schema dict)
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": declaration.name,
                            "description": declaration.description,
                            "parameters": declaration.parameters # Assumes direct compatibility
                        }
                    }
                    openai_tools.append(tool_dict)
            except Exception as e:
                log.error(f"Error converting tool '{name}' to OpenAI format: {e}", exc_info=True)
                
        return openai_tools if openai_tools else None 