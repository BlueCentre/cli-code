"""
Gemini model integration for the CLI tool.
"""

# Standard Library
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import google.api_core.exceptions

# Third-party Libraries
import google.generativeai as genai
import google.generativeai.types as genai_types
import questionary
import rich
from google.api_core.exceptions import GoogleAPIError
from google.generativeai.types import HarmBlockThreshold, HarmCategory
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

    # --- generate method remains largely the same, ensure signature matches base ---
    def generate(self, prompt: str) -> Optional[str]:
        logging.info(f"Agent Loop - Processing prompt: '{prompt[:100]}...' using model '{self.current_model_name}'")

        # Early checks and validations
        # Check for empty prompts
        if not prompt or prompt.strip() == "":
            log.warning("Empty prompt provided to generate()")
            return "Error: Cannot process empty prompt. Please provide a valid input."

        # Check if model is initialized
        if not self.model:
            log.error("Model is not initialized")
            return "Error: Model is not initialized. Please try again or check your API key."

        # Add initial user prompt to history
        self.add_to_history({"role": "user", "parts": [prompt]})

        original_user_prompt = prompt
        if prompt.startswith("/"):
            command = prompt.split()[0].lower()
            # Handle commands like /compact here eventually
            if command == "/exit":
                logging.info(f"Handled command: {command}")
                return None  # Exit command will be handled by the caller
            elif command == "/help":
                logging.info(f"Handled command: {command}")
                return self._get_help_text()  # Return help text

        # === Step 1: Get Initial Context ===
        orientation_context = self._get_initial_context()

        # === Step 2: Prepare Initial User Turn ===
        # Combine orientation with the actual user request
        turn_input_prompt = f"{orientation_context}\nUser request: {original_user_prompt}"

        # Add this combined input to the PERSISTENT history
        self.add_to_history({"role": "user", "parts": [turn_input_prompt]})
        # === START DEBUG LOGGING ===
        log.debug(f"Prepared turn_input_prompt (sent to LLM):\n---\n{turn_input_prompt}\n---")
        # === END DEBUG LOGGING ===
        self._manage_context_window()  # Truncate *before* sending the first request

        iteration_count = 0
        task_completed = False
        final_summary = ""
        last_text_response = "No response generated."  # Fallback text

        try:
            # === Agent Loop with Status Animation ===
            with self.console.status("[bold green]Thinking...[/bold green]", spinner="dots") as status:
                while iteration_count < MAX_AGENT_ITERATIONS and not task_completed:
                    iteration_count += 1
                    log.info(f"--- Agent Loop Iteration: {iteration_count} ---")
                    log.debug(f"Current History: {self.history}")  # DEBUG

                    status.update("[bold green]Thinking...[/bold green]")

                    try:
                        # Ensure history is not empty before sending
                        if not self.history:
                            log.error("Agent history became empty unexpectedly.")
                            return "Error: Agent history is empty."

                        llm_response = self.model.generate_content(
                            self.history,
                            generation_config=self.generation_config,
                            tools=[self.gemini_tools] if self.gemini_tools else None,
                            safety_settings=SAFETY_SETTINGS,
                            request_options={"timeout": 600},  # Timeout for potentially long tool calls
                        )
                        log.debug(f"LLM Response (Iter {iteration_count}): {llm_response}")  # DEBUG

                        # --- Response Processing ---
                        if not llm_response.candidates:
                            log.error(f"LLM response had no candidates. Prompt Feedback: {llm_response.prompt_feedback}")
                            if llm_response.prompt_feedback and llm_response.prompt_feedback.block_reason:
                                block_reason = llm_response.prompt_feedback.block_reason.name
                                # Provide more specific feedback if blocked
                                return f"Error: Prompt was blocked by API. Reason: {block_reason}"
                            else:
                                return "Error: Empty response received from LLM (no candidates)."

                        response_candidate = llm_response.candidates[0]
                        log.debug(f"-- Processing Candidate {response_candidate.index} --")  # DEBUG

                        # <<< NEW: Prioritize STOP Reason Check >>>
                        if response_candidate.finish_reason == 1:  # STOP
                            log.info("STOP finish reason received. Checking for final text.")
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

                            if final_text.strip(): # If there WAS text content with the STOP
                                log.info("Model stopped with final text content.")
                                final_summary = final_text.strip()
                                task_completed = True
                                break # Exit loop immediately on STOP with text
                            else:
                                # log.warning("Model stopped (finish_reason=STOP) but provided no text content. Letting loop continue or finish naturally.") # Removed warning
                                # Do NOT set final_summary here
                                # Do NOT set task_completed = True here
                                # Do NOT break here - let the loop potentially timeout or handle unexpected exit later
                                pass # Continue processing other parts or finish loop iteration

                        # <<< END NEW STOP CHECK >>>

                        # --- Start Part Processing ---
                        function_call_part_to_execute = None
                        text_response_buffer = ""
                        processed_function_call_in_turn = False

                        # --- ADD CHECK for content being None ---
                        if response_candidate.content is None:
                            log.warning(f"Response candidate {response_candidate.index} had no content object.")
                            # Treat same as having no parts - check finish reason
                            if response_candidate.finish_reason == 2:  # MAX_TOKENS
                                final_summary = "(Response terminated due to maximum token limit)"
                                task_completed = True
                            elif response_candidate.finish_reason != 1:  # Not STOP
                                final_summary = f"(Response candidate {response_candidate.index} finished unexpectedly: {response_candidate.finish_reason} with no content)"
                                task_completed = True
                            # If STOP or UNSPECIFIED, let loop continue / potentially time out if nothing else happens

                        elif not response_candidate.content.parts:
                            # Existing check for empty parts list
                            log.warning(
                                f"Response candidate {response_candidate.index} had content but no parts. Finish Reason: {response_candidate.finish_reason}"
                            )
                            if response_candidate.finish_reason == 2:  # MAX_TOKENS
                                final_summary = "(Response terminated due to maximum token limit)"
                                task_completed = True
                            elif response_candidate.finish_reason != 1:  # Not STOP
                                final_summary = f"(Response candidate {response_candidate.index} finished unexpectedly: {response_candidate.finish_reason} with no parts)"
                                task_completed = True
                            pass
                        else:
                            # Process parts if they exist
                            for part in response_candidate.content.parts:
                                log.debug(f"-- Processing Part: {part} (Type: {type(part)}) --")
                                if (
                                    hasattr(part, "function_call")
                                    and part.function_call
                                    and not processed_function_call_in_turn
                                ):
                                    log.info(f"LLM requested Function Call part: {part.function_call}")  # Simple log
                                    self.add_to_history({"role": "model", "parts": [part]})
                                    self._manage_context_window()
                                    function_call_part_to_execute = part  # Store the part itself
                                    processed_function_call_in_turn = True
                                elif hasattr(part, "text") and part.text:  # Ensure this block is correct
                                    llm_text = part.text
                                    log.info(f"LLM returned text part (Iter {iteration_count}): {llm_text[:100]}...")
                                    text_response_buffer += llm_text + "\n"  # Append text
                                    self.add_to_history({"role": "model", "parts": [part]})
                                    self._manage_context_window()
                                else:
                                    # Handle unexpected parts if necessary, ensure logging is appropriate
                                    log.warning(f"LLM returned unexpected response part (Iter {iteration_count}): {part}")
                                    # Decide if unexpected parts should be added to history
                                    self.add_to_history({"role": "model", "parts": [part]})
                                    self._manage_context_window()

                        # --- Start Decision Block ---
                        if function_call_part_to_execute:
                            # Extract name and args here + type check
                            function_call = function_call_part_to_execute.function_call
                            tool_name_obj = function_call.name
                            tool_args = dict(function_call.args) if function_call.args else {}

                            # Explicitly check type of extracted name object
                            if isinstance(tool_name_obj, str):
                                tool_name_str = tool_name_obj
                            else:
                                tool_name_str = str(tool_name_obj)
                                log.warning(
                                    f"Tool name object was not a string (type: {type(tool_name_obj)}), converted using str() to: '{tool_name_str}'"
                                )

                            log.info(f"Executing tool: {tool_name_str} with args: {tool_args}")

                            try:
                                status.update(f"[bold blue]Running tool: {tool_name_str}...[/bold blue]")

                                tool_instance = get_tool(tool_name_str)
                                if not tool_instance:
                                    # log.error(f"[Tool Exec] Tool '{tool_name_str}' not found instance: {tool_instance}") # REMOVE DEBUG
                                    result_for_history = {"error": f"Error: Tool '{tool_name_str}' not found."}
                                else:
                                    # log.debug(f"[Tool Exec] Tool instance found: {tool_instance}") # REMOVE DEBUG
                                    if tool_name_str == "task_complete":
                                        summary = tool_args.get("summary", "Task completed.")
                                        log.info(f"Task complete requested by LLM: {summary}")
                                        final_summary = summary
                                        task_completed = True
                                        # log.debug("[Tool Exec] Task complete logic executed.") # REMOVE DEBUG
                                        # Append simulated tool response using dict structure
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
                                        # log.debug("[Tool Exec] Appended task_complete ack to history.") # REMOVE DEBUG
                                        break
                                    else:
                                        # log.debug(f"[Tool Exec] Preparing to execute {tool_name_str} with args: {tool_args}") # REMOVE DEBUG

                                        # --- Confirmation Check ---
                                        if tool_name_str in TOOLS_REQUIRING_CONFIRMATION:
                                            log.info(f"Requesting confirmation for sensitive tool: {tool_name_str}")
                                            confirm_msg = f"Allow the AI to execute the '{tool_name_str}' command with arguments: {tool_args}?"
                                            try:
                                                # Use ask() which returns True, False, or None (for cancel)
                                                confirmation = questionary.confirm(
                                                    confirm_msg,
                                                    auto_enter=False,  # Require explicit confirmation
                                                    default=False,  # Default to no if user just hits enter
                                                ).ask()

                                                if confirmation is not True:  # Handles False and None (cancel)
                                                    log.warning(
                                                        f"User rejected or cancelled execution of tool: {tool_name_str}"
                                                    )
                                                    rejection_message = f"User rejected execution of tool: {tool_name_str}"
                                                    # Add rejection message to history for the LLM
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
                                                    continue  # Skip execution and proceed to next iteration
                                            except Exception as confirm_err:
                                                log.error(
                                                    f"Error during confirmation prompt for {tool_name_str}: {confirm_err}",
                                                    exc_info=True,
                                                )
                                                # Treat confirmation error as rejection for safety
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
                                                continue  # Skip execution

                                            log.info(f"User confirmed execution for tool: {tool_name_str}")
                                        # --- End Confirmation Check ---

                                        tool_result = tool_instance.execute(**tool_args)
                                        # log.debug(f"[Tool Exec] Finished executing {tool_name_str}. Result: {tool_result}") # REMOVE DEBUG

                                        # Format result for history
                                        if isinstance(tool_result, dict):
                                            result_for_history = tool_result
                                        elif isinstance(tool_result, str):
                                            result_for_history = {"output": tool_result}
                                        else:
                                            result_for_history = {"output": str(tool_result)}
                                            log.warning(
                                                f"Tool {tool_name_str} returned non-dict/str result: {type(tool_result)}. Converting to string."
                                            )

                                        # Append tool response using dict structure
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
                                        # log.debug("[Tool Exec] Appended tool result to history.") # REMOVE DEBUG
                                        self._manage_context_window()

                                        # Update status back after tool execution (before next iteration)
                                        status.update("[bold green]Thinking...[/bold green]")
                                        continue

                            except Exception as e:
                                error_message = f"Error: Tool execution error with {tool_name_str}: {e}"
                                log.exception(f"[Tool Exec] Exception caught: {error_message}")  # Keep exception log
                                # <<< NEW: Set summary and break loop >>>
                                final_summary = error_message
                                task_completed = True
                                break  # Exit loop to handle final output consistently
                                # <<< END NEW >>>

                            # function_call_part_to_execute = None # Clear the stored part - Now unreachable due to return
                            # continue # Continue loop after processing function call - Now unreachable due to return

                        elif task_completed:
                            log.info("Task completed flag is set. Finalizing.")
                            break
                        elif text_response_buffer:
                            log.info(
                                f"Text response buffer has content ('{text_response_buffer.strip()}'). Finalizing."
                            )  # Log buffer content
                            final_summary = text_response_buffer
                            break  # Exit loop
                        else:
                            # This case means the LLM response had no text AND no function call processed in this iteration.
                            log.warning(
                                f"Agent loop iteration {iteration_count}: No actionable parts found or processed. Continuing."
                            )
                            # Check finish reason if no parts were actionable using integer values
                            # Assuming FINISH_REASON_STOP = 1, FINISH_REASON_UNSPECIFIED = 0
                            if response_candidate.finish_reason != 1 and response_candidate.finish_reason != 0:
                                log.warning(
                                    f"Response candidate {response_candidate.index} finished unexpectedly ({response_candidate.finish_reason}) with no actionable parts. Exiting loop."
                                )
                                final_summary = f"(Agent loop ended due to unexpected finish reason: {response_candidate.finish_reason} with no actionable parts)"
                                task_completed = True
                            pass

                    except StopIteration:
                        # This occurs when mock side_effect is exhausted
                        log.warning("StopIteration caught, likely end of mock side_effect sequence.")
                        # Decide what to do - often means the planned interaction finished.
                        # If a final summary wasn't set by text_response_buffer, maybe use last known text?
                        if not final_summary:
                            log.warning("Loop ended due to StopIteration without a final summary set.")
                            # Optionally find last text from history here if needed
                            # For this test, breaking might be sufficient if text_response_buffer worked.
                            final_summary = "(Loop ended due to StopIteration)"  # Fallback summary
                        task_completed = True  # Ensure loop terminates
                        break  # Exit loop

                    except google.api_core.exceptions.ResourceExhausted as quota_error:
                        # Log full details at debug level
                        log.debug(f"Full quota error details: {quota_error}") # Log full details for debugging
                        # Check if we are already using the fallback
                        if self.current_model_name == FALLBACK_MODEL:
                            log.error("Quota exceeded even for the fallback model. Cannot proceed.")
                            self.console.print(
                                "[bold red]API quota exceeded for primary and fallback models. Please check your plan/billing.[/bold red]"
                            )
                            # Clean history before returning
                            if self.history[-1]["role"] == "user":
                                self.history.pop()
                            return "Error: API quota exceeded for primary and fallback models."
                        else:
                            log.info(f"Switching to fallback model: {FALLBACK_MODEL}")
                            status.update(f"[bold yellow]Switching to fallback model: {FALLBACK_MODEL}...[/bold yellow]")
                            self.console.print(
                                f"[bold yellow]Quota limit reached for {self.current_model_name}. Switching to fallback model ({FALLBACK_MODEL})...[/bold yellow]"
                            )
                            self.current_model_name = FALLBACK_MODEL
                            try:
                                self._initialize_model_instance()
                                log.info(
                                    f"Successfully switched to and initialized fallback model: {self.current_model_name}"
                                )
                                # Important: Clear the last model response (which caused the error) before retrying
                                if self.history[-1]["role"] == "model":
                                    last_part = self.history[-1]["parts"][0]
                                    # Only pop if it was a failed function call attempt or empty text response leading to error
                                    if (
                                        hasattr(last_part, "function_call")
                                        or not hasattr(last_part, "text")
                                        or not last_part.text
                                    ):
                                        self.history.pop()
                                        log.debug("Removed last model part before retrying with fallback.")
                                continue  # Retry the current loop iteration with the new model
                            except Exception as fallback_init_error:
                                log.error(
                                    f"Failed to initialize fallback model '{FALLBACK_MODEL}': {fallback_init_error}",
                                    exc_info=True,
                                )
                                self.console.print(
                                    f"[bold red]Error switching to fallback model: {fallback_init_error}[/bold red]"
                                )
                                if self.history[-1]["role"] == "user":
                                    self.history.pop()
                                return "Error: Failed to initialize fallback model after quota error."

                    except Exception as generation_error:
                        # This handles other errors during the generate_content call or loop logic
                        log.error(f"Error during Agent Loop: {generation_error}", exc_info=True)
                        # Ensure status stops on error
                        # The 'with' statement handles this automatically
                        # Clean history
                        if self.history[-1]["role"] == "user":
                            self.history.pop()
                        return f"Error during agent processing: {generation_error}"

            # === End Agent Loop ===
            # The 'with' statement ensures status stops here

            # === Handle Final Output ===
            if task_completed and final_summary:
                log.info("Agent loop finished. Returning final summary.")
                # Cleanup internal tags if needed (using a hypothetical method)
                # cleaned_summary = self._cleanup_internal_tags(final_summary)
                return final_summary.strip()  # Return the summary from task_complete or final text
            elif iteration_count >= MAX_AGENT_ITERATIONS:
                log.warning(f"Agent loop terminated after reaching max iterations ({MAX_AGENT_ITERATIONS}).")
                # Try to get the last *text* response the model generated, even if it wanted to call a function after
                last_model_response_text = self._find_last_model_text(self.history)
                timeout_message = f"(Task exceeded max iterations ({MAX_AGENT_ITERATIONS}). Last text from model was: {last_model_response_text})"
                return timeout_message.strip()
            else:
                # This case should be less likely now
                log.error("Agent loop exited unexpectedly.")
                last_model_response_text = self._find_last_model_text(self.history)
                return f"(Agent loop finished unexpectedly. Last model text: {last_model_response_text})"

        except Exception as e:
            log.error(f"Error during Agent Loop: {str(e)}", exc_info=True)
            # Ensure status stops on outer error
            # The 'with' statement handles this automatically
            return f"An unexpected error occurred during the agent process: {str(e)}"

    def generate_without_init(self, prompt):
        """Generate a response without client init (to handle error cases)."""
        if not prompt or prompt.strip() == "":
            return "Error: Empty prompt provided. Please enter a valid command or query."

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
                if isinstance(item["parts"][0], str):
                    return item["parts"][0]
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
            self.history = [] # Should not happen if initialized correctly
        log.info("Gemini history cleared.")

    # --- Help Text Generator ---
    def _get_help_text(self) -> str:
        # Implementation of _get_help_text method
        pass
