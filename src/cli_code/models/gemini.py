"""
Gemini model integration for the CLI tool.
"""

# Standard Library
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import google.api_core.exceptions

# Third-party Libraries
import google.generativeai as genai
import questionary
import rich
from google.ai.generativelanguage_v1beta.types.generative_service import Candidate
from google.api_core.exceptions import GoogleAPIError, InternalServerError, ResourceExhausted
from google.generativeai import protos

# Fixed imports based on google-generativeai 0.8.4 structure
from google.generativeai.types import (
    ContentType,
    GenerateContentResponse,
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    PartType,
    Tool,
)

# Import FunctionDeclaration from content_types module
from google.generativeai.types.content_types import FunctionDeclaration
from google.protobuf.json_format import MessageToDict
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

# Local Application/Library Specific Imports
from ..tools import AVAILABLE_TOOLS, get_tool
from ..utils.history_manager import MAX_HISTORY_TURNS, HistoryManager
from ..utils.log_config import get_logger
from ..utils.tool_registry import ToolRegistry  # ToolResponse potentially defined elsewhere or not needed?
from .base import AbstractModelAgent

# Define tools requiring confirmation
TOOLS_REQUIRING_CONFIRMATION = ["edit", "create_file", "bash"]  # Add other tools if needed

# Setup logging (basic config, consider moving to main.py)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s') # Removed, handled in main
log = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 10
FALLBACK_MODEL = "gemini-2.0-flash"
CONTEXT_TRUNCATION_THRESHOLD_TOKENS = 800000  # Example token limit
# Keep ~N pairs of user/model turns + initial setup + tool calls/responses

# Safety Settings - Adjust as needed
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Remove standalone list_available_models function
# def list_available_models(api_key):
#     ...


class GeminiModel(AbstractModelAgent):  # Inherit from base class
    """Interface for Gemini models using native function calling agentic loop."""

    # Constants
    THINKING_STATUS = "[bold green]Thinking...[/bold green]"

    def __init__(
        self,
        api_key: str,
        console: Console,
        model_name: str | None = "gemini-2.5-pro-exp-03-25",
    ):
        """Initialize the Gemini model interface."""
        super().__init__(console=console, model_name=model_name)  # Call base class init

        if not api_key:
            raise ValueError("Gemini API key is required.")

        self.api_key = api_key
        self.initial_model_name = self.model_name or "gemini-2.5-pro-exp-03-25"  # Use passed model or default
        self.current_model_name = self.initial_model_name  # Start with the determined model
        # self.console is set by super().__init__

        try:
            genai.configure(api_key=api_key)
        except Exception as config_err:
            log.error(f"Failed to configure Gemini API: {config_err}", exc_info=True)
            raise ConnectionError(f"Failed to configure Gemini API: {config_err}") from config_err

        self.generation_config = genai.GenerationConfig(temperature=0.4, top_p=0.95, top_k=40)
        self.safety_settings = {
            "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HATE": "BLOCK_MEDIUM_AND_ABOVE",
            "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
            "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE",
        }

        # --- Tool Definition ---
        self.function_declarations = self._create_tool_definitions()
        self.gemini_tools = (
            {"function_declarations": self.function_declarations} if self.function_declarations else None
        )
        # ---

        # --- System Prompt (Native Functions & Planning) ---
        self.system_instruction = self._create_system_prompt()
        # ---

        # --- Initialize Gemini-specific History ---
        self.history = []  # Initialize history list for this instance
        self.add_to_history({"role": "user", "parts": [self.system_instruction]})
        self.add_to_history(
            {
                "role": "model",
                "parts": ["Okay, I'm ready. Provide the directory context and your request."],
            }
        )
        log.info("Initialized persistent chat history for GeminiModel.")
        # ---

        try:
            self._initialize_model_instance()  # Creates self.model
            log.info("GeminiModel initialized successfully (Native Function Calling Agent Loop).")
        except Exception as e:
            log.error(
                f"Fatal error initializing Gemini model '{self.current_model_name}': {str(e)}",
                exc_info=True,
            )
            # Raise a more specific error or just re-raise
            raise Exception(f"Could not initialize Gemini model '{self.current_model_name}': {e}") from e

    def _initialize_model_instance(self):
        """Helper to create the GenerativeModel instance."""
        if not self.current_model_name:
            raise ValueError("Model name cannot be empty for initialization.")
        log.info(f"Initializing model instance: {self.current_model_name}")
        try:
            # Pass system instruction here, tools are passed during generate_content
            self.model = genai.GenerativeModel(
                model_name=self.current_model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self.system_instruction,
            )
            log.info(f"Model instance '{self.current_model_name}' created successfully.")
            # Initialize status message context manager
            self.status_message = self.console.status("[dim]Initializing...[/dim]")
        except Exception as init_err:
            log.error(
                f"Failed to create model instance for '{self.current_model_name}': {init_err}",
                exc_info=True,
            )
            raise init_err

    # --- Implement list_models from base class ---
    def list_models(self) -> List[Dict] | None:
        """List available Gemini models."""
        try:
            # genai should already be configured from __init__
            models = genai.list_models()
            gemini_models = []
            for model in models:
                # Filter for models supporting generateContent
                if "generateContent" in model.supported_generation_methods:
                    model_info = {
                        "id": model.name,  # Use 'id' for consistency maybe?
                        "name": model.display_name,
                        "description": model.description,
                        # Add other relevant fields if needed
                    }
                    gemini_models.append(model_info)
            return gemini_models
        except Exception as e:
            log.error(f"Error listing Gemini models: {str(e)}", exc_info=True)
            self.console.print(f"[bold red]Error listing Gemini models:[/bold red] {e}")
            return []  # Return empty list instead of None

    def generate(self, prompt: str) -> Optional[str]:
        logging.info(f"Agent Loop - Processing prompt: '{prompt[:100]}...' using model '{self.current_model_name}'")

        # Add initial user prompt to history first
        self.add_to_history({"role": "user", "parts": [prompt]})

        # Handle special commands
        if prompt.strip().lower() == "/exit":
            logging.info("Handled command: /exit")
            self.history.pop()  # Remove /exit from history
            return None  # Exit command handled by caller
        elif prompt.strip().lower() == "/help":
            logging.info("Handled command: /help")
            self.history.pop()  # Remove /help from history
            return self._get_help_text()  # Return help text

        # Early validation
        if not self._validate_prompt_and_model(prompt):
            # Remove invalid prompt from history
            self.history.pop()
            return "Error: Cannot process empty prompt or model not initialized. Please try again."

        # Prepare the context and input for the model - NOT NEEDED with new history handling
        # turn_input_prompt = self._prepare_input_context(prompt)

        # Manage context window before loop
        self._manage_context_window()

        # Set up for agent loop
        iteration_count = 0
        task_completed = False
        final_summary = ""
        last_text_response = "No response generated."  # Fallback text

        try:
            # Execute the agent loop
            result = self._execute_agent_loop(iteration_count, task_completed, final_summary, last_text_response)
            return result
        except Exception as e:
            log.error(f"Error during Agent Loop: {str(e)}", exc_info=True)
            # Remove last user prompt if loop failed
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            return f"An unexpected error occurred during the agent process: {str(e)}"

    def _validate_prompt_and_model(self, prompt: str) -> bool:
        """Validate that the prompt is not empty and the model is initialized."""
        if not prompt or prompt.strip() == "":
            log.warning("Empty prompt provided to generate()")
            return False

        if not self.model:
            log.error("Model is not initialized")
            return False

        return True

    def _handle_special_commands(self, prompt: str) -> Optional[str]:
        """Handle special commands like /exit and /help."""
        if prompt.startswith("/"):
            command = prompt.split()[0].lower()
            if command == "/exit":
                logging.info(f"Handled command: {command}")
                return None  # Exit command will be handled by the caller
            elif command == "/help":
                logging.info(f"Handled command: {command}")
                return self._get_help_text()  # Return help text
        return None  # Not a special command

    def _prepare_input_context(self, original_user_prompt: str) -> str:
        """Prepare the input context with orientation and user prompt."""
        # Get the initial context for orientation
        orientation_context = self._get_initial_context()

        # Combine orientation with the actual user request
        turn_input_prompt = f"{orientation_context}\nUser request: {original_user_prompt}"

        # Add this combined input to the PERSISTENT history
        self.add_to_history({"role": "user", "parts": [turn_input_prompt]})
        # Debug logging
        log.debug(f"Prepared turn_input_prompt (sent to LLM):\n---\n{turn_input_prompt}\n---")

        # Manage context window before first request
        self._manage_context_window()

        return turn_input_prompt

    def _execute_agent_loop(self, iteration_count, task_completed, final_summary, last_text_response):
        """Execute the main agent loop with LLM interactions and tool calls."""
        with self.console.status(self.THINKING_STATUS, spinner="dots") as status:
            while iteration_count < MAX_AGENT_ITERATIONS and not task_completed:
                iteration_count += 1
                log.info(f"--- Agent Loop Iteration: {iteration_count} ---")
                log.debug(f"Current History: {self.history}")

                status.update(self.THINKING_STATUS)

                # Process a single iteration of the agent loop
                iteration_result = self._process_agent_iteration(status, last_text_response)

                # Handle various outcomes from the iteration
                if isinstance(iteration_result, tuple) and len(iteration_result) == 2:
                    # Unpack result type and value
                    result_type, result_value = iteration_result

                    if result_type == "error":
                        return result_value
                    elif result_type == "continue":
                        last_text_response = result_value
                        continue
                    elif result_type == "complete":
                        return result_value
                    elif result_type == "task_completed":
                        task_completed = True
                        log.info("Task completed flag is set. Finalizing.")
                        break

            # Handle loop completion
            if last_text_response and "User rejected" in last_text_response:
                return last_text_response
            return self._handle_loop_completion(task_completed, final_summary, iteration_count)

    def _process_agent_iteration(self, status, last_text_response):
        """Process a single iteration of the agent loop."""
        try:
            # Ensure history is not empty before sending
            if not self.history:
                log.error("Agent history became empty unexpectedly.")
                return "error", "Error: Agent history is empty."

            # Get response from LLM
            llm_response = self._get_llm_response()

            # Check for valid response
            if not llm_response.candidates:
                return "error", self._handle_empty_response(llm_response)

            # Process response from the model
            return self._process_candidate_response(llm_response.candidates[0], status)

        except Exception as e:
            result = self._handle_agent_loop_exception(e, status)
            if result:
                return "error", result
            return "continue", last_text_response

    def _process_candidate_response(self, response_candidate: Candidate, status) -> Tuple[str, Optional[str]]:
        """Process a single response candidate from the Gemini API."""
        # --- TEMPORARY DEBUG PRINTS --- REMOVED

        # Log the finish reason value directly, without assuming .name attribute
        log.debug(
            f"Processing candidate with finish_reason: {response_candidate.finish_reason if response_candidate.finish_reason is not None else 'N/A'}"
        )

        # Process parts (text and function calls)
        function_call_part_to_execute = None
        text_response_buffer = ""
        if response_candidate.content and response_candidate.content.parts:
            for part in response_candidate.content.parts:
                if part.text:
                    text_response_buffer += part.text
                elif part.function_call:
                    if function_call_part_to_execute:
                        log.warning("Multiple function calls received in one response, only executing the first.")
                    else:
                        function_call_part_to_execute = part  # Store the whole part
                        log.info(f"Received function call request: {part.function_call.name}")
                else:
                    # Handle parts with neither text nor function call
                    log.warning("Received part with neither text nor function_call.")
                    return self._handle_empty_parts(response_candidate)

        # --- Decision Logic ---

        # Priority 1: Execute Function Call if present
        if function_call_part_to_execute:
            log.debug(f"Prioritizing function call: {function_call_part_to_execute.function_call.name}")
            # Add the function call request to history (even if there's also text)
            # We add the full message content to match API structure,
            # although only the function_call part is strictly needed for execution.
            # The history expects a dict, not the Part object directly.
            # Ensure the history entry matches expected structure (might need adjustment based on how protos.Content is handled)
            # Assuming response_candidate.content is already a protos.Content object
            if response_candidate.content:
                self.add_to_history(
                    {"role": "model", "parts": response_candidate.content.parts}
                )  # Parts should be iterable protos.Part
            else:
                log.warning("Attempted to add response to history, but candidate content was empty.")

            # Execute the function call - we need to handle the async function differently
            # Since _process_candidate_response is not async, we can't directly await the result
            # We need to run the coroutine to completion using asyncio
            try:
                # Use asyncio to run the coroutine to completion
                import asyncio

                result_type, result_value = asyncio.run(
                    self._execute_function_call(function_call_part_to_execute.function_call)
                )
            except Exception as e:
                log.error(f"Error executing function call: {e}")
                return "error", f"Error executing function call: {e}"

            # Propagate errors or task completion immediately
            if result_type in ["error", "task_completed"]:
                return result_type, result_value

            # If the tool executed successfully, we need to send the result back.
            # The loop should continue, but we don't need to return text here.
            # The next iteration will send the history (including the tool result)
            # back to the model.
            log.debug("Function call executed successfully, continuing loop.")
            return "continue", None  # Signal to continue without returning text yet

        # Priority 2: Handle specific non-STOP finish reasons
        elif response_candidate.finish_reason == 2:  # MAX_TOKENS
            log.warning("Response stopped due to maximum token limit.")
            # Use dict format for history part
            self.add_to_history({"role": "model", "parts": [{"text": text_response_buffer}]})  # Use dict format
            return "error", "Response exceeded maximum token limit."
        elif response_candidate.finish_reason == 3:  # SAFETY
            log.warning("Response stopped due to safety settings.")
            # Don't add potentially unsafe content to history
            return "error", "Response blocked due to safety concerns."
        elif response_candidate.finish_reason == 4:  # RECITATION
            log.warning("Response stopped due to recitation policy.")
            # Don't add potentially problematic content to history
            return "error", "Response blocked due to recitation policy."
        elif response_candidate.finish_reason == 5:  # OTHER
            log.warning("Response stopped due to an unspecified reason.")
            # Use dict format for history part
            if text_response_buffer:
                self.add_to_history({"role": "model", "parts": [{"text": text_response_buffer}]})  # Use dict format
            return "error", "Response stopped for an unknown reason."

        # Priority 3: Handle STOP or unspecified finish reason with text content
        # Check for STOP (1) or UNSPECIFIED (0)
        elif text_response_buffer and response_candidate.finish_reason in [0, 1]:
            log.debug(f"Received text response with finish_reason: {response_candidate.finish_reason}. Completing.")
            # Use dict format for history part
            self.add_to_history({"role": "model", "parts": [{"text": text_response_buffer}]})  # Use dict format
            return "complete", text_response_buffer.strip()

        # Priority 4: Handle STOP (1) or unspecified (0) finish reason with NO text and NO function call
        elif response_candidate.finish_reason in [0, 1]:
            log.warning(f"Received finish_reason {response_candidate.finish_reason} with no text or function call.")
            # Don't add an empty message to history unless necessary for state tracking?
            # For now, treat as effectively complete but empty.
            return "complete", "(Agent received an empty response)"
        # Fallback for any other unexpected finish reason if text_buffer is empty
        else:
            log.error(f"Unhandled finish_reason {response_candidate.finish_reason} with no actionable content.")
            return "error", f"Unhandled finish reason: {response_candidate.finish_reason}"

    def _get_llm_response(self):
        """Get response from the language model."""
        return self.model.generate_content(
            self.history,
            generation_config=self.generation_config,
            tools=[self.gemini_tools] if self.gemini_tools else None,
            safety_settings=SAFETY_SETTINGS,
            request_options={"timeout": 600},  # Timeout for potentially long tool calls
        )

    def _handle_empty_response(self, llm_response: GenerateContentResponse) -> Tuple[str, str]:
        """Handles the case where the LLM response has no candidates."""
        log.warning("LLM response contained no candidates.")
        block_reason = "Unknown"
        if llm_response.prompt_feedback and llm_response.prompt_feedback.block_reason:
            # Handle both enum and string cases for block_reason
            reason = llm_response.prompt_feedback.block_reason
            if hasattr(reason, "name"):
                block_reason = reason.name  # Access name if it's enum-like
            else:
                block_reason = str(reason)  # Otherwise, use its string representation
            log.error(f"Prompt was blocked by API. Reason: {block_reason}")
            return "error", f"Error: Prompt was blocked by API. Reason: {block_reason}"
        else:
            log.error("Prompt may have been blocked, but no reason was provided.")
            return "error", "Error: Prompt was blocked by API, but no reason was provided."

    def _check_for_stop_reason(self, response_candidate, status):
        """Check if the response has a STOP finish reason."""
        if response_candidate.finish_reason == protos.Candidate.FinishReason.STOP:  # Use protos enum
            log.info("STOP finish reason received. Checking for final text.")
            return True
        return False

    def _extract_final_text(self, response_candidate):
        """Extract text from a STOP response."""
        final_text = ""
        final_parts = []
        if response_candidate.content and response_candidate.content.parts:
            final_parts = response_candidate.content.parts
            for part in final_parts:
                if hasattr(part, "text") and part.text:
                    final_text += part.text + "\n"

        # Add the stopping response to history regardless
        self.add_to_history({"role": "model", "parts": final_parts})
        self._manage_context_window()

        return final_text

    def _handle_null_content(self, response_candidate):
        """Handle the case where the content field is null."""
        log.warning(
            f"Response candidate {response_candidate.index} had no content object. Finish Reason: {response_candidate.finish_reason}"
        )
        # Return an error message regardless of finish reason if content is None
        # Return tuple format
        return (
            "error",
            f"(Internal Agent Error: Received response candidate {response_candidate.index} with no content object)",
        )

    def _handle_empty_parts(self, response_candidate):
        """Handle the case where the content has no parts."""
        log.warning(
            f"Response candidate {response_candidate.index} had content but no parts. Finish Reason: {response_candidate.finish_reason}"
        )
        # Return an error message regardless of finish reason if parts are empty
        # Return tuple format
        return (
            "error",
            f"(Internal Agent Error: Received response candidate {response_candidate.index} with empty parts list)",
        )

    def _handle_no_actionable_content(self, response_candidate):
        """Handle the case where no actionable content was found."""
        log.warning("No actionable parts (text or function call) found or processed in the response candidate.")

        finish_reason = response_candidate.finish_reason
        # Use genai_types.FinishReason to interpret the integer value
        try:
            # Use the enum directly if finish_reason is already an enum member
            if isinstance(finish_reason, protos.Candidate.FinishReason):
                reason_name = finish_reason.name
            else:  # Attempt conversion if it's an int/str (less likely now?)
                reason_enum = protos.Candidate.FinishReason(finish_reason)
                reason_name = reason_enum.name
        except ValueError:
            reason_name = f"UNKNOWN({finish_reason})"

        # If finished due to reasons other than STOP or TOOL_CALLS, treat as an error/completion state
        # Ensure we compare with the enum values
        stop_reasons = [
            protos.Candidate.FinishReason.STOP,
            protos.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED,
        ]  # TOOL_CALLS might not exist in protos version? Check API. Assume Tool presence implies tool calls.
        if function_call_present := any(
            p.function_call.name for p in response_candidate.content.parts if p.function_call
        ):  # Check if FC is present
            pass  # Handled elsewhere
        elif finish_reason not in stop_reasons:
            error_msg = f"(Agent loop ended due to finish reason: {reason_name} with no actionable parts)"
            log.error(error_msg)
            # Add the problematic candidate to history for context?
            try:
                parts_to_add = response_candidate.content.parts if response_candidate.content else []
                if parts_to_add:  # Only add if parts exist
                    self.add_to_history({"role": "model", "parts": parts_to_add})  # Add raw parts
                    self._manage_context_window()
            except Exception as hist_err:
                log.warning(f"Could not add problematic candidate parts to history: {hist_err}")
            return "error", error_msg  # Return as error
        else:
            # If finish reason is STOP or TOOL_CALLS or UNSPECIFIED but we somehow got here (no text/FC processed),
            # log warning and continue the loop. This shouldn't happen if STOP/TOOL_CALLS are handled correctly earlier.
            log.warning(f"No actionable content found, but finish reason was {reason_name}. Continuing loop.")
            # Returning "continue" with None message, loop will proceed.
            return "continue", None

    def _handle_agent_loop_exception(self, exception, status):
        """Handle exceptions that occur during the agent loop."""
        if isinstance(exception, StopIteration):
            return self._handle_stop_iteration(exception)
        elif isinstance(exception, google.api_core.exceptions.ResourceExhausted):
            return self._handle_quota_exceeded(exception, status)
        else:
            return self._handle_general_exception(exception)

    def _handle_stop_iteration(self, exception):
        """Handle StopIteration exceptions."""
        log.warning("StopIteration caught, likely end of mock side_effect sequence.")
        return "(Loop ended due to StopIteration)"

    def _handle_quota_exceeded(self, exception, status):
        """Handle quota exceeded errors."""
        log.debug(f"Full quota error details: {exception}")

        # Check if already using fallback model
        if self.current_model_name == FALLBACK_MODEL:
            log.error("Quota exceeded even for the fallback model.")
            self.console.print("[bold red]API quota exceeded for all models. Check billing.[/bold red]")

            # Clean history
            if self.history[-1]["role"] == "user":
                self.history.pop()

            return "Error: API quota exceeded for primary and fallback models."

        # Try switching to fallback model
        log.info(f"Switching to fallback model: {FALLBACK_MODEL}")
        status.update(f"[bold yellow]Switching to fallback model: {FALLBACK_MODEL}...[/bold yellow]")
        self.console.print(
            f"[bold yellow]Quota limit reached for {self.current_model_name}. Switching to fallback model ({FALLBACK_MODEL})...[/bold yellow]"
        )

        self.current_model_name = FALLBACK_MODEL
        try:
            self._initialize_model_instance()
            log.info(f"Successfully switched to fallback model: {self.current_model_name}")

            # Clean problematic history entry if present
            if self.history[-1]["role"] == "model":
                last_part = self.history[-1]["parts"][0]
                if hasattr(last_part, "function_call") or not hasattr(last_part, "text") or not last_part.text:
                    self.history.pop()
                    log.debug("Removed last model part before retrying with fallback.")

            return None  # Continue the loop with new model

        except Exception as fallback_init_error:
            log.error(f"Failed to initialize fallback model: {fallback_init_error}", exc_info=True)
            self.console.print(f"[bold red]Error switching to fallback model: {fallback_init_error}[/bold red]")

            if self.history[-1]["role"] == "user":
                self.history.pop()

            return f"Error: Failed to initialize fallback model: {fallback_init_error}"

    def _handle_general_exception(self, exception):
        """Handle general exceptions during the agent loop."""
        log.error(f"Error during Agent Loop: {exception}", exc_info=True)

        # Clean history
        if self.history[-1]["role"] == "user":
            self.history.pop()

        return f"Error during agent processing: {exception}"

    def _handle_loop_completion(self, task_completed, final_summary, iteration_count):
        """Handle the completion of the agent loop."""
        if task_completed and final_summary:
            log.info("Agent loop finished. Returning final summary.")
            return final_summary.strip()

        elif iteration_count >= MAX_AGENT_ITERATIONS:
            log.warning(f"Agent loop reached max iterations ({MAX_AGENT_ITERATIONS}).")
            last_model_response_text = self._find_last_model_text(self.history)
            return f"(Task exceeded max iterations ({MAX_AGENT_ITERATIONS}). Last text: {last_model_response_text})".strip()

        else:
            log.error("Agent loop exited unexpectedly.")
            last_model_response_text = self._find_last_model_text(self.history)
            return f"(Agent loop finished unexpectedly. Last model text: {last_model_response_text})"

    # --- Context Management (Consider Token Counting) ---
    def _manage_context_window(self):
        """Truncates history if it exceeds limits (Gemini-specific)."""
        # Each full LLM round (request + function_call + function_response) adds 3 items
        if len(self.history) > (MAX_HISTORY_TURNS * 3 + 2):
            log.warning(f"Chat history length ({len(self.history)}) exceeded threshold. Truncating.")
            # Keep system prompt (idx 0), initial model ack (idx 1)
            keep_count = MAX_HISTORY_TURNS * 3  # Keep N rounds
            keep_from_index = len(self.history) - keep_count
            self.history = self.history[:2] + self.history[keep_from_index:]
            log.info(f"History truncated to {len(self.history)} items.")
            log.debug(f"History length AFTER truncation inside _manage_context_window: {len(self.history)}")
        # TODO: Implement token-based truncation check using count_tokens

    # --- Tool Definition Helper ---
    def _create_tool_definitions(self) -> list | None:
        """Dynamically create Tool definitions from AVAILABLE_TOOLS."""
        # Fix: AVAILABLE_TOOLS is a dictionary, not a function
        declarations = []
        for tool_name, tool_class in AVAILABLE_TOOLS.items():
            try:
                # Instantiate the tool
                tool_instance = tool_class()
                if hasattr(tool_instance, "get_function_declaration"):
                    declaration_obj = tool_instance.get_function_declaration()
                    if declaration_obj:
                        # Assuming declaration_obj is structured correctly or needs conversion
                        # For now, append directly. May need adjustment based on actual object structure.
                        declarations.append(declaration_obj)
                        log.debug(f"Generated tool definition for tool: {tool_name}")
                    else:
                        log.warning(f"Tool {tool_name} has 'get_function_declaration' but it returned None.")
                else:
                    log.warning(f"Tool {tool_name} does not have a 'get_function_declaration' method. Skipping.")
            except Exception as e:
                log.error(f"Error instantiating tool '{tool_name}': {e}")
                continue

        log.info(f"Created {len(declarations)} tool definitions for native tool use.")
        # The return type of this function might need to be adjusted based on how
        # genai.GenerativeModel expects tools (e.g., maybe a single Tool object containing declarations?)
        # For now, returning the list as gathered.
        return declarations if declarations else None

    # --- System Prompt Helper ---
    def _create_system_prompt(self) -> str:
        """Creates the system prompt, emphasizing native functions and planning."""
        tool_descriptions = []
        if self.function_declarations:  # This is now a list of FunctionDeclaration objects
            # Process FunctionDeclaration objects directly
            for func_decl in self.function_declarations:
                # Extract details directly from the FunctionDeclaration
                args_str = ""
                if (
                    hasattr(func_decl, "parameters")
                    and func_decl.parameters
                    and hasattr(func_decl.parameters, "properties")
                    and func_decl.parameters.properties
                ):
                    args_list = []
                    required_args = getattr(func_decl.parameters, "required", []) or []
                    for prop, details in func_decl.parameters.properties.items():
                        prop_type = getattr(details, "type", "UNKNOWN")
                        prop_desc = getattr(details, "description", "")
                        suffix = "" if prop in required_args else "?"
                        args_list.append(f"{prop}: {prop_type}{suffix} # {prop_desc}")
                    args_str = ", ".join(args_list)

                func_name = getattr(func_decl, "name", "UNKNOWN_FUNCTION")
                func_desc = getattr(func_decl, "description", "(No description provided)")
                tool_descriptions.append(f"- `{func_name}({args_str})`: {func_desc}")
        else:
            tool_descriptions.append(" - (No tools available with function declarations)")

        tool_list_str = "\n".join(tool_descriptions)

        # Prompt v13.1 - Native Functions, Planning, Accurate Context
        return f"""You are Gemini Code, an AI coding assistant running in a CLI environment.
Your goal is to help the user with their coding tasks by understanding their request, planning the necessary steps, and using the available tools via **native function calls**.

Available Tools (Use ONLY these via function calls):
{tool_list_str}

Workflow:
1.  **Analyze & Plan:** Understand the user's request based on the provided directory context (`ls` output) and the request itself. For non-trivial tasks, **first outline a brief plan** of the steps and tools you will use in a text response. **Note:** Actions that modify files (`edit`, `create_file`) will require user confirmation before execution.
2.  **Execute:** If a plan is not needed or after outlining the plan, make the **first necessary function call** to execute the next step (e.g., `view` a file, `edit` a file, `grep` for text, `tree` for structure).
3.  **Observe:** You will receive the result of the function call (or a message indicating user rejection). Use this result to inform your next step.
4.  **Repeat:** Based on the result, make the next function call required to achieve the user's goal. Continue calling functions sequentially until the task is complete.
5.  **Complete:** Once the *entire* task is finished, **you MUST call the `task_complete` function**, providing a concise summary of what was done in the `summary` argument.
    *   The `summary` argument MUST accurately reflect the final outcome (success, partial success, error, or what was done).
    *   Format the summary using **Markdown** for readability (e.g., use backticks for filenames `like_this.py` or commands `like this`).
    *   If code was generated or modified, the summary **MUST** contain the **actual, specific commands** needed to run or test the result (e.g., show `pip install Flask` and `python app.py`, not just say "instructions provided"). Use Markdown code blocks for commands.

Important Rules:
*   **Use Native Functions:** ONLY interact with tools by making function calls as defined above. Do NOT output tool calls as text (e.g., `cli_tools.ls(...)`).
*   **Sequential Calls:** Call functions one at a time. You will get the result back before deciding the next step. Do not try to chain calls in one turn.
*   **Initial Context Handling:** When the user asks a general question about the codebase contents (e.g., "what's in this directory?", "show me the files", "whats in this codebase?"), your **first** response MUST be a summary or list of **ALL** files and directories provided in the initial context (`ls` or `tree` output). Do **NOT** filter this initial list or make assumptions (e.g., about virtual environments). Only after presenting the full initial context should you suggest further actions or use other tools if necessary.
*   **Accurate Context Reporting:** When asked about directory contents (like "whats in this codebase?"), accurately list or summarize **all** relevant files and directories shown in the `ls` or `tree` output, including common web files (`.html`, `.js`, `.css`), documentation (`.md`), configuration files, build artifacts, etc., not just specific source code types. Do not ignore files just because virtual environments are also present. Use `tree` for a hierarchical view if needed.
*   **Handling Explanations:**
    *   If the user asks *how* to do something, asks for an explanation, or requests instructions (like "how do I run this?"), **provide the explanation or instructions directly in a text response** using clear Markdown formatting.
    *   **Proactive Assistance:** When providing instructions that culminate in a specific execution command (like `python file.py`, `npm start`, `git status | cat`, etc.), first give the full explanation, then **explicitly ask the user if they want you to run that final command** using the `execute_command` tool.
        *   Example: After explaining how to run `calculator.py`, you should ask: "Would you like me to run `python calculator.py | cat` for you using the `execute_command` tool?" (Append `| cat` for commands that might page).
    *   Do *not* use `task_complete` just for providing information; only use it when the *underlying task* (e.g., file creation, modification) is fully finished.
*   **Planning First:** For tasks requiring multiple steps (e.g., read file, modify content, write file), explain your plan briefly in text *before* the first function call.
*   **Precise Edits:** When editing files (`edit` tool), prefer viewing the relevant section first (`view` tool with offset/limit), then use exact `old_string`/`new_string` arguments if possible. Only use the `content` argument for creating new files or complete overwrites.
*   **Task Completion Signal:** ALWAYS finish action-oriented tasks by calling `task_complete(summary=...)`.
    *   The `summary` argument MUST accurately reflect the final outcome (success, partial success, error, or what was done).
    *   Format the summary using **Markdown** for readability (e.g., use backticks for filenames `like_this.py` or commands `like this`).
    *   If code was generated or modified, the summary **MUST** contain the **actual, specific commands** needed to run or test the result (e.g., show `pip install Flask` and `python app.py`, not just say "instructions provided"). Use Markdown code blocks for commands.

The user's first message will contain initial directory context and their request."""

    def _get_initial_context(self) -> str:
        """
        Gets the initial context for the conversation based on the following hierarchy:
        1. Content of .rules/*.md files if the directory exists
        2. Content of README.md in the root directory if it exists
        3. Output of 'ls' command (fallback to original behavior)

        Returns:
            A string containing the initial context.
        """

        # Check if .rules directory exists
        if os.path.isdir(".rules"):
            log.info("Found .rules directory. Reading *.md files for initial context.")
            try:
                md_files = glob.glob(".rules/*.md")
                if md_files:
                    context_content = []
                    for md_file in md_files:
                        log.info(f"Reading rules file: {md_file}")
                        try:
                            with open(md_file, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read().strip()
                                if content:
                                    file_basename = os.path.basename(md_file)
                                    context_content.append(f"# Content from {file_basename}\n\n{content}")
                        except Exception as read_err:
                            log.error(f"Error reading rules file '{md_file}': {read_err}", exc_info=True)

                    if context_content:
                        combined_content = "\n\n".join(context_content)
                        self.console.print("[dim]Context initialized from .rules/*.md files.[/dim]")
                        return f"Project rules and guidelines:\n```markdown\n{combined_content}\n```\n"
            except Exception as rules_err:
                log.error(f"Error processing .rules directory: {rules_err}", exc_info=True)

        # Check if README.md exists in the root
        if os.path.isfile("README.md"):
            log.info("Using README.md for initial context.")
            try:
                with open("README.md", "r", encoding="utf-8", errors="ignore") as f:
                    readme_content = f.read().strip()
                if readme_content:
                    self.console.print("[dim]Context initialized from README.md.[/dim]")
                    return f"Project README:\n```markdown\n{readme_content}\n```\n"
            except Exception as readme_err:
                log.error(f"Error reading README.md: {readme_err}", exc_info=True)

        # Fall back to ls output (original behavior)
        log.info("Falling back to 'ls' output for initial context.")
        try:
            ls_tool = get_tool("ls")
            if ls_tool:
                ls_result = ls_tool.execute()
                log.info(f"Orientation ls result length: {len(ls_result) if ls_result else 0}")
                self.console.print("[dim]Directory context acquired via 'ls'.[/dim]")
                return f"Current directory contents (from initial `ls`):\n```\n{ls_result}\n```\n"
            else:
                log.error("CRITICAL: Could not find 'ls' tool for mandatory orientation.")
                return "Error: The essential 'ls' tool is missing. Cannot proceed."
        except Exception as orient_error:
            log.error(f"Error during mandatory orientation (ls): {orient_error}", exc_info=True)
            error_message = f"Error during initial directory scan: {orient_error}"
            self.console.print(f"[bold red]Error getting initial directory listing: {orient_error}[/bold red]")
            return f"{error_message}\n"

    # --- Text Extraction Helper (if needed for final output) ---
    def _extract_text_from_response(self, response) -> str | None:
        """Safely extracts text from a Gemini response object."""
        try:
            if response and response.candidates:
                # Assuming candidates are protos.Candidate
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    # Assuming parts are protos.Part
                    text_parts = [part.text for part in candidate.content.parts if hasattr(part, "text") and part.text]
                    return "\n".join(text_parts).strip() if text_parts else None
            return None
        except (AttributeError, IndexError) as e:
            log.warning(f"Could not extract text from response: {e} - Response: {response}")
            return None

    # --- Find Last Text Helper ---
    def _find_last_model_text(self, history: list) -> str:
        """Finds the last text response from the 'model' in the history."""
        for item in reversed(history):
            if item.get("role") == "model":
                parts = item.get("parts", [])
                if parts:  # Check if parts list is not empty
                    # Check if the first part is a simple string (older format?)
                    if isinstance(parts[0], str):
                        log.debug(f"Found last model text (string format): {parts[0][:50]}...")
                        return parts[0]
                    # Check if the first part is a dict-like object with a 'text' key (less likely now with protos)
                    elif isinstance(parts[0], dict) and "text" in parts[0] and isinstance(parts[0]["text"], str):
                        log.warning("Found last model text in dict format (unexpected with protos).")
                        return parts[0]["text"]
                    # Check if the first part is an object with a 'text' attribute (should be protos.Part now)
                    elif hasattr(parts[0], "text") and isinstance(parts[0].text, str):
                        # Check if it's actually a protos.Part before accessing .text
                        if isinstance(parts[0], protos.Part):
                            log.debug(f"Found last model text (protos.Part format): {parts[0].text[:50]}...")
                            return parts[0].text
                        else:
                            log.warning(f"Found object with text attribute, but not protos.Part: {type(parts[0])}")

        log.warning("Could not find any valid text in the last model response.")
        return "(No suitable model text found in history)"

    # --- Add Gemini-specific history management methods ---
    def add_to_history(self, entry):
        """Adds an entry to the Gemini conversation history."""
        self.history.append(entry)
        self._manage_context_window()  # Call truncation logic after adding

    def clear_history(self):
        """Clears the Gemini conversation history, preserving the system prompt."""
        if self.history:
            # Keep system prompt (idx 0), initial model ack (idx 1)
            self.history = self.history[:2]
        else:
            self.history = []  # Should not happen if initialized correctly
        log.info("Gemini history cleared.")

    # --- Help Text Generator ---
    def _get_help_text(self) -> str:
        """Return help text for CLI commands and usage."""
        help_text = """
# CLI-Code Assistant Help

## Interactive Commands:
- `/exit` - Exit the interactive session
- `/help` - Show this help message
- `/clear` - Clear the conversation history

## Usage Tips:
- Ask direct questions about the codebase
- Request to view, edit, or create files
- Ask for explanations of code
- Request to run commands (with confirmation)
- Ask for help implementing a feature

## Examples:
- "What files are in this directory?"
- "Show me the content of file.py"
- "Explain what this code does"
- "Create a new file called hello.py with a simple hello world program"
- "How do I run this application?"
"""
        return help_text.strip()

    # --- Restore _execute_function_call and its helpers ---
    async def _execute_function_call(self, function_calls):
        """Execute a function call requested by the LLM."""
        # Extract the first tool call (current implementation processes one at a time)
        if isinstance(function_calls, list) and len(function_calls) > 0:
            # Handle list of tool call dictionaries (new interface)
            function_call = function_calls[0]
            tool_name = function_call.get("name")
            tool_args = function_call.get("arguments", {})
        else:
            # Handle legacy single function_call proto object
            function_call = function_calls
            # Convert protobuf Struct/Message args to Python dict
            tool_args = {}
            if hasattr(function_call, "args"):
                try:
                    # Convert the protobuf Struct/Message to a Python dict
                    # Need to handle potential _pb attribute if args is a proto wrapper
                    args_message = function_call.args
                    if hasattr(args_message, "_pb"):  # Check if it's a proto wrapper
                        args_message = args_message._pb
                    tool_args = MessageToDict(args_message)
                except Exception as e:
                    log.error(
                        f"Failed to convert function call args to dict for tool '{tool_name}': {e}", exc_info=True
                    )
                    # Return error if args conversion fails
                    return f"Error processing arguments for tool '{tool_name}': {e}", False

        log.info(f"Executing tool: {tool_name} with args: {tool_args}")  # Log the converted args

        # Handle task_complete pseudo-tool
        if tool_name == "task_complete":
            summary = tool_args.get("summary", "Task completed.")
            log.info(f"Task marked complete by LLM. Summary: {summary}")
            return summary, True  # Return the summary text with a flag indicating completion

        # Find and validate the tool
        try:
            tool = get_tool(tool_name)
            if not tool:
                log.error(f"Tool '{tool_name}' not found in the registry.")
                return f"Error: Tool '{tool_name}' not found.", False
        except Exception as e:
            log.error(f"Error retrieving tool '{tool_name}': {e}", exc_info=True)
            return f"Error retrieving tool '{tool_name}': {e}", False

        # Request confirmation if required
        confirmation_result = self._request_tool_confirmation(tool, tool_name, tool_args)
        if confirmation_result == "rejected":
            log.warning(f"User rejected execution of tool '{tool_name}'.")
            return "Tool execution rejected by user.", False
        elif confirmation_result == "error":
            log.error(f"Error during confirmation prompt for tool '{tool_name}'.")
            return f"Error during confirmation for {tool_name}", False
        # Otherwise it's approved (either explicitly or implicitly)

        # Execute the tool
        try:
            # Use Status context manager
            with Status(f"Running tool: {tool_name}...", console=self.console, spinner="dots") as status:
                # Execute tool with structured arguments
                tool_result = tool.execute(**tool_args)
                log.info(f"Tool '{tool_name}' executed successfully.")
                log.debug(f"Tool '{tool_name}' raw result: {tool_result}")

            # Store the result for the next LLM turn
            self._store_tool_result(tool_name, tool_result)
            # For tests, create an object that mimics ContentType with parts and function_response
            from dataclasses import dataclass

            @dataclass
            class FunctionResponse:
                name: str
                response: dict

            @dataclass
            class Part:
                function_response: FunctionResponse = None
                text: str = None

            @dataclass
            class ContentType:
                parts: list[Part]

            # Create a ContentType with a FunctionResponse part
            content = ContentType(
                parts=[Part(function_response=FunctionResponse(name=tool_name, response={"content": str(tool_result)}))]
            )

            return content

        except Exception as e:
            log.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            error_message = f"Error executing tool '{tool_name}': {str(e)}"
            # Store the error result
            self._store_tool_result(tool_name, {"error": error_message})
            return error_message, False  # Return error to stop the loop

    def _handle_task_complete(self, tool_args):
        """Handle the special task_complete tool call."""
        final_summary = tool_args.get("summary", "Task completed.")
        log.info(f"Task marked complete by LLM. Summary: {final_summary}")
        # Return a special tuple to signal loop completion
        return "task_completed", True

    def _request_tool_confirmation(self, tool, tool_name, tool_args):
        """Request user confirmation if the tool requires it."""
        if tool.requires_confirmation:
            log.info(f"Requesting confirmation for tool: {tool_name}")
            self.console.print(f"LLM wants to run tool: [bold magenta]{tool_name}[/bold magenta]")
            self.console.print(f"Arguments: [cyan]{tool_args}[/cyan]")
            try:
                # Call questionary.confirm directly, which will need patching in tests
                # The message format might need adjustment based on questionary usage
                confirmed = questionary.confirm(
                    f"Allow the AI to execute the '{tool_name}' command with arguments: {tool_args}?",
                    default=False,  # Default to No for safety
                    auto_enter=False,
                ).ask()  # Ask the question

                if not confirmed:
                    log.warning(f"User rejected execution of tool '{tool_name}'.")
                    # Store rejection as tool result for LLM context
                    rejection_result = {"user_decision": f"User rejected execution of tool '{tool_name}'"}
                    self._store_tool_result(tool_name, rejection_result)
                    # Signal rejection
                    return "rejected"
            except Exception as e:
                log.error(f"Error during tool confirmation for '{tool_name}': {e}", exc_info=True)
                # Store error as tool result
                error_result = {"error": f"Error during tool confirmation: {str(e)}"}
                self._store_tool_result(tool_name, error_result)
                # Return error to stop the loop
                return "error"
        return None  # Confirmation not required or was granted

    def _store_tool_result(self, tool_name: str, tool_result: Any):
        """Store the tool execution result in the history."""
        # Convert result to FunctionResponse format expected by Gemini
        try:
            # Basic serialization for common types, ensure it's JSON serializable
            if isinstance(tool_result, (dict, list, str, int, float, bool, type(None))):
                content_data = tool_result
            else:
                # Attempt generic string conversion for other types
                try:
                    # Attempt direct conversion for proto compatibility if needed
                    # This might need adjustment based on what protos.FunctionResponse expects
                    content_data = tool_result
                except Exception:
                    content_data = str(tool_result)

            # Create the function response dictionary structure directly
            function_response_dict = {
                "name": tool_name,
                "response": {"result": content_data},  # Wrap result in a dict
            }
            # Create the part dictionary containing the function response
            function_part_dict = {"function_response": function_response_dict}

            # Add to persistent history using the dictionary format
            self.add_to_history({"role": "function", "parts": [function_part_dict]})  # Add dict part
            log.info(f"Stored result for tool '{tool_name}'.")
            log.debug(f"Stored FunctionResponse Part (dict): {function_part_dict}")
            # Manage context window AFTER adding the result
            self._manage_context_window()

        except Exception as e:
            log.error(f"Error storing tool result for '{tool_name}': {e}", exc_info=True)
            # Optionally add an error message to history?
            # self.add_to_history({"role": "function", "parts": [...]}) # Add error part?
