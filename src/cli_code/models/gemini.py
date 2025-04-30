"""
Gemini model integration for the CLI tool.
"""

# Standard Library
import glob
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import google.api_core.exceptions

# Third-party Libraries
import google.generativeai as genai
import google.generativeai.types as genai_types
import questionary
import rich
from google.api_core.exceptions import GoogleAPIError
from google.generativeai.types import FunctionDeclaration, HarmBlockThreshold, HarmCategory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Local Application/Library Specific Imports
from ..tools import AVAILABLE_TOOLS, get_tool
from .base import AbstractModelAgent

# Define tools requiring confirmation
TOOLS_REQUIRING_CONFIRMATION = ["edit", "create_file", "bash"]  # Add other tools if needed

# Setup logging (basic config, consider moving to main.py)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s') # Removed, handled in main
log = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 10
FALLBACK_MODEL = "gemini-2.0-flash"
CONTEXT_TRUNCATION_THRESHOLD_TOKENS = 800000  # Example token limit
MAX_HISTORY_TURNS = 20  # Keep ~N pairs of user/model turns + initial setup + tool calls/responses

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
        """Generate a response using the Gemini model with function calling capabilities."""
        logging.info(f"Agent Loop - Processing prompt: '{prompt[:100]}...' using model '{self.current_model_name}'")

        # --- Start: Handle special commands early ---
        # Handle special commands *before* adding initial prompt to history or preparing context
        command_response = self._handle_special_commands(prompt)
        if command_response is not None:  # Help command returned text
            return command_response
        elif prompt.strip().lower() == "/exit":  # Check specifically for /exit after handling
            return None  # Return None immediately for exit
        # --- End: Handle special commands early ---

        # Early validation
        if not self._validate_prompt_and_model(prompt):
            return "Error: Cannot process empty prompt or model not initialized. Please try again."

        # Add initial user prompt to history (only if not a handled command)
        self.add_to_history({"role": "user", "parts": [prompt]})

        # Prepare the context and input for the model
        turn_input_prompt = self._prepare_input_context(prompt)

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
                        final_summary = result_value  # Store the summary
                        # Handle potential None in log message
                        summary_log_part = f"{final_summary[:100]}..." if final_summary else "None"
                        log.info(f"Task completed flag is set. Finalizing with summary: {summary_log_part}")
                        break

            # Handle loop completion (max iterations or task complete)
            # Add check for rejection message here before handling normal completion
            if last_text_response and "User rejected" in last_text_response:
                log.info("Agent loop finished after user rejection.")
                return last_text_response
            # Pass the captured final_summary here
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

    def _process_candidate_response(self, response_candidate, status):
        """Process a response candidate from the LLM."""
        log.debug(f"-- Processing Candidate {response_candidate.index} --")

        # Check for STOP finish reason first
        if self._check_for_stop_reason(response_candidate, status):
            final_text = self._extract_final_text(response_candidate)
            if final_text.strip():
                return "complete", final_text.strip()
            # If stop reason but no text, fall through to process content anyway

        # Process the response content
        result = self._process_response_content(response_candidate, status)

        # Check the result type
        if result is not None:
            # Check if the result is a rejection message
            if isinstance(result, str) and "User rejected" in result:
                log.info("User rejection detected, continuing agent loop.")
                return "continue", result  # Keep loop going, pass rejection message back
            else:
                # Otherwise, assume it's a final text response or task completion signal
                log.info("Final text response or completion signal received.")
                return "complete", result

        # If result is None, it means either no actionable content initially,
        # OR a tool was executed successfully and the loop should continue.
        log.info("No text response or completion signal, continuing agent loop.")
        # Return a generic continue message if the tool succeeded but didn't return text
        return "continue", "Tool executed, continue loop."

    def _get_llm_response(self):
        """Get response from the language model."""
        return self.model.generate_content(
            self.history,
            generation_config=self.generation_config,
            tools=[self.gemini_tools] if self.gemini_tools else None,
            safety_settings=SAFETY_SETTINGS,
            request_options={"timeout": 600},  # Timeout for potentially long tool calls
        )

    def _handle_empty_response(self, llm_response):
        """Handle the case where the LLM response has no candidates."""
        log.error(f"LLM response had no candidates. Prompt Feedback: {llm_response.prompt_feedback}")
        if llm_response.prompt_feedback and llm_response.prompt_feedback.block_reason:
            block_reason = llm_response.prompt_feedback.block_reason.name
            # Provide more specific feedback if blocked
            return f"Error: Prompt was blocked by API. Reason: {block_reason}"
        else:
            return "Error: Empty response received from LLM (no candidates)."

    def _check_for_stop_reason(self, response_candidate, status):
        """Check if the response has a STOP finish reason."""
        if response_candidate.finish_reason == 1:  # STOP
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

    def _process_response_content(self, response_candidate, status):
        """Process the content of a response candidate."""
        # Initialize tracking variables
        text_response_buffer = ""

        # Check for content being None
        if response_candidate.content is None:
            return self._handle_null_content(response_candidate)

        # Check for empty parts list
        if not response_candidate.content.parts:
            return self._handle_empty_parts(response_candidate)

        # Process parts: Execute the *first* function call encountered.
        processed_function_call_in_turn = False
        for part in response_candidate.content.parts:
            log.debug(f"-- Processing Part: {part} (Type: {type(part)}) --")

            # Handle function call part
            if hasattr(part, "function_call") and part.function_call:
                log.info(f"LLM requested Function Call part: {part.function_call}")
                # Add the function call request to history
                self.add_to_history({"role": "model", "parts": [part]})
                self._manage_context_window()

                # Execute the function call immediately
                execution_result = self._execute_function_call(part, status)
                processed_function_call_in_turn = True

                # If execution returned a message (error, rejection, task complete), return it
                if execution_result is not None:
                    log.info(f"Function call execution returned result: {execution_result}")
                    # Check specifically for rejection to ensure it bubbles up immediately
                    if isinstance(execution_result, str) and "User rejected" in execution_result:
                        return execution_result  # Return rejection message immediately
                    # Otherwise, return the result (could be error or task complete summary)
                    return execution_result
                else:
                    # Tool executed successfully, continue agent loop (return None from this method)
                    log.info("Function call executed successfully, continuing agent loop.")
                    return None  # Signal outer loop to continue

            # Handle text part (accumulate text found before any function call)
            elif hasattr(part, "text") and part.text:
                llm_text = part.text
                log.info(f"LLM returned text part: {llm_text[:100]}...")
                text_response_buffer += llm_text + "\n"
                # Add text part to history
                self.add_to_history({"role": "model", "parts": [part]})
                self._manage_context_window()

            # Handle unexpected part types
            else:
                log.warning(f"LLM returned unexpected response part: {part}")
                # Add unexpected part to history anyway
                self.add_to_history({"role": "model", "parts": [part]})
                self._manage_context_window()

        # --- Loop finished ---

        # If we processed a function call that returned None, we already returned None above.
        # If we exit the loop without processing a function call:
        if not processed_function_call_in_turn:
            if text_response_buffer:
                log.info(f"No function call executed. Returning accumulated text: '{text_response_buffer.strip()}'")
                return text_response_buffer.strip()
            else:
                # Handle case with no actionable content at all
                return self._handle_no_actionable_content(response_candidate)

        # Should technically be unreachable if a function call was processed
        return None  # Default return if something unexpected happens

    def _handle_null_content(self, response_candidate):
        """Handle the case where the content field is null."""
        log.warning(f"Response candidate {response_candidate.index} had no content object.")
        if response_candidate.finish_reason == 2:  # MAX_TOKENS
            return "(Response terminated due to maximum token limit)"
        elif response_candidate.finish_reason != 1:  # Not STOP
            return f"(Response candidate {response_candidate.index} finished unexpectedly: {response_candidate.finish_reason} with no content)"
        return None

    def _handle_empty_parts(self, response_candidate):
        """Handle the case where the content has no parts."""
        log.warning(
            f"Response candidate {response_candidate.index} had content but no parts. Finish Reason: {response_candidate.finish_reason}"
        )
        if response_candidate.finish_reason == 2:  # MAX_TOKENS
            return "(Response terminated due to maximum token limit)"
        elif response_candidate.finish_reason != 1:  # Not STOP
            return f"(Response candidate {response_candidate.index} finished unexpectedly: {response_candidate.finish_reason} with no parts)"
        return None

    def _execute_function_call(self, function_call_part, status):
        """Execute a function call from the LLM."""
        # Extract function details
        function_call = function_call_part.function_call
        tool_name_obj = function_call.name
        tool_args = dict(function_call.args) if function_call.args else {}

        # Validate tool name
        if isinstance(tool_name_obj, str):
            tool_name_str = tool_name_obj
        else:
            tool_name_str = str(tool_name_obj)
            log.warning(f"Tool name was not a string, converted to: '{tool_name_str}'")

        log.info(f"Executing tool: {tool_name_str} with args: {tool_args}")

        try:
            status.update(f"[bold blue]Running tool: {tool_name_str}...[/bold blue]")

            # Get the tool instance
            tool_instance = get_tool(tool_name_str)
            if not tool_instance:
                result_for_history = {"error": f"Error: Tool '{tool_name_str}' not found."}
                return f"Error: Tool '{tool_name_str}' not found."

            # Handle task_complete tool specially
            if tool_name_str == "task_complete":
                return self._handle_task_complete(tool_name_str, tool_args)

            # Handle tools requiring confirmation
            if tool_name_str in TOOLS_REQUIRING_CONFIRMATION:
                confirmation_result = self._request_tool_confirmation(tool_name_str, tool_args)
                if confirmation_result:
                    return confirmation_result

            # Execute the tool
            tool_result = tool_instance.execute(**tool_args)

            # Format and store the result
            self._store_tool_result(tool_name_str, tool_result)

            # Update status back to thinking
            status.update(self.THINKING_STATUS)

        except Exception as e:
            error_message = f"Error: Tool execution error with {tool_name_str}: {e}"
            log.exception(f"[Tool Exec] Exception caught: {error_message}")
            return error_message

        return None  # Continue the loop

    def _handle_task_complete(self, tool_name_str, tool_args):
        """Handle the task_complete tool call."""
        summary = tool_args.get("summary", "Task completed.")
        log.info(f"Task complete requested by LLM: {summary}")

        # Add acknowledgment to history
        self.history.append(
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": tool_name_str,
                            "response": {"status": "acknowledged"},
                        }
                    }
                ],
            }
        )

        return summary.strip()

    def _request_tool_confirmation(self, tool_name_str, tool_args):
        """Request user confirmation for sensitive tools."""
        log.info(f"Requesting confirmation for sensitive tool: {tool_name_str}")
        confirm_msg = f"Allow the AI to execute the '{tool_name_str}' command with arguments: {tool_args}?"

        try:
            confirmation = questionary.confirm(
                confirm_msg,
                auto_enter=False,
                default=False,
            ).ask()

            if confirmation is not True:
                log.warning(f"User rejected execution of tool: {tool_name_str}")
                rejection_message = f"User rejected execution of tool: {tool_name_str}"

                # Add rejection to history
                self.history.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": tool_name_str,
                                    "response": {
                                        "status": "rejected",
                                        "message": rejection_message,
                                    },
                                }
                            }
                        ],
                    }
                )
                self._manage_context_window()

                # Return the rejection message which will be caught by _execute_function_call
                return f"User rejected the proposed {tool_name_str} operation on {tool_args.get('file_path', 'unknown file')}"

        except KeyboardInterrupt:
            # Handle user cancellation explicitly
            log.warning("User cancelled tool confirmation.")
            cancellation_message = "User cancelled tool confirmation."
            # Add cancellation to history (similar to rejection/error)
            self.history.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": tool_name_str,
                                "response": {
                                    "status": "cancelled",
                                    "message": cancellation_message,
                                },
                            }
                        }
                    ],
                }
            )
            self._manage_context_window()
            return cancellation_message  # Return specific cancellation message

        except Exception as confirm_err:
            log.error(f"Error during confirmation: {confirm_err}", exc_info=True)

            # Add error to history
            self.history.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": tool_name_str,
                                "response": {
                                    "status": "error",
                                    "message": f"Error during confirmation: {confirm_err}",
                                },
                            }
                        }
                    ],
                }
            )
            self._manage_context_window()
            return f"Error during confirmation: {confirm_err}"

        return None  # Confirmation successful, continue execution

    def _store_tool_result(self, tool_name_str, tool_result):
        """Format and store a tool execution result in history."""
        # Format the result for history
        if isinstance(tool_result, dict):
            result_for_history = tool_result
        elif isinstance(tool_result, str):
            result_for_history = {"output": tool_result}
        else:
            result_for_history = {"output": str(tool_result)}
            log.warning(f"Tool {tool_name_str} returned non-dict/str result. Converting to string.")

        # Add to history
        self.history.append(
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": tool_name_str,
                            "response": result_for_history,
                        }
                    }
                ],
            }
        )
        self._manage_context_window()

    def _handle_no_actionable_content(self, response_candidate):
        """Handle the case where no actionable content was found."""
        log.warning("No actionable parts found or processed.")

        # Check finish reason for unexpected values
        if response_candidate.finish_reason != 1 and response_candidate.finish_reason != 0:
            log.warning(
                f"Response finished unexpectedly ({response_candidate.finish_reason}) with no actionable parts."
            )
            return f"(Agent loop ended due to unexpected finish reason: {response_candidate.finish_reason} with no actionable parts)"

        return None  # Continue the loop

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
                # Handle potential multi-part responses if ever needed, for now assume text is in the first part
                if response.candidates[0].content and response.candidates[0].content.parts:
                    text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, "text")]
                    return "\n".join(text_parts).strip() if text_parts else None
            return None
        except (AttributeError, IndexError) as e:
            log.warning(f"Could not extract text from response: {e} - Response: {response}")
            return None

    # --- Find Last Text Helper ---
    def _find_last_model_text(self, history: list) -> str:
        for item in reversed(history):
            if item["role"] == "model":
                parts = item.get("parts")  # Safely get parts
                # Check if parts exists, is a list, and is not empty
                if parts and isinstance(parts, list) and len(parts) > 0:
                    # Check if the first part is a string
                    if isinstance(parts[0], str):
                        return parts[0]
                    # Optional: Check if first part has a 'text' attribute
                    elif hasattr(parts[0], "text") and parts[0].text:
                        return parts[0].text.strip()
        return "No text found in history"

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
