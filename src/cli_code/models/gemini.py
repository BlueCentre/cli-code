"""
Gemini model integration for the CLI tool.
"""

# Standard Library
import glob
import logging
import os
from typing import Dict, List

import google.api_core.exceptions

# Third-party Libraries
import google.generativeai as genai
import google.generativeai.types as genai_types
import questionary
import rich
from rich.console import Console

# from rich.panel import Panel # Remove unused import
# Local Application/Library Specific Imports
from ..tools import AVAILABLE_TOOLS, get_tool
from .base import AbstractModelAgent

# Setup logging (basic config, consider moving to main.py)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s') # Removed, handled in main
log = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 10
FALLBACK_MODEL = "gemini-1.5-pro-latest"
CONTEXT_TRUNCATION_THRESHOLD_TOKENS = 800000  # Example token limit
MAX_HISTORY_TURNS = 20  # Keep ~N pairs of user/model turns + initial setup + tool calls/responses

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

        self.generation_config = genai.types.GenerationConfig(temperature=0.4, top_p=0.95, top_k=40)
        self.safety_settings = {
            "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HATE": "BLOCK_MEDIUM_AND_ABOVE",
            "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
            "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE",
        }

        # --- Tool Definition ---
        self.function_declarations = self._create_tool_definitions()
        self.gemini_tools = (
            genai_types.Tool(function_declarations=self.function_declarations) if self.function_declarations else None
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
            return None  # Indicate failure

    # --- generate method remains largely the same, ensure signature matches base ---
    def generate(self, prompt: str) -> str | None:
        logging.info(f"Agent Loop - Processing prompt: '{prompt[:100]}...' using model '{self.current_model_name}'")
        original_user_prompt = prompt
        if prompt.startswith("/"):
            command = prompt.split()[0].lower()
            # Handle commands like /compact here eventually
            if command in ["/exit", "/help"]:
                logging.info(f"Handled command: {command}")
                return None  # Or return specific help text

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
        final_summary = None
        last_text_response = "No response generated."  # Fallback text

        try:
            while iteration_count < MAX_AGENT_ITERATIONS:
                iteration_count += 1
                logging.info(f"Agent Loop Iteration {iteration_count}/{MAX_AGENT_ITERATIONS}")

                # === Call LLM with History and Tools ===
                llm_response = None
                try:
                    logging.info(
                        f"Sending request to LLM ({self.current_model_name}). History length: {len(self.history)} turns."
                    )
                    # === ADD STATUS FOR LLM CALL ===
                    with self.console.status(
                        f"[yellow]Assistant thinking ({self.current_model_name})...",
                        spinner="dots",
                    ):
                        # Pass the available tools to the generate_content call
                        llm_response = self.model.generate_content(
                            self.history,
                            generation_config=self.generation_config,
                            tools=[self.gemini_tools] if self.gemini_tools else None,
                        )
                    # === END STATUS ===

                    # === START DEBUG LOGGING ===
                    log.debug(f"RAW Gemini Response Object (Iter {iteration_count}): {llm_response}")
                    # === END DEBUG LOGGING ===

                    # Extract the response part (candidate)
                    # Add checks for empty candidates or parts
                    if not llm_response.candidates:
                        log.error(f"LLM response had no candidates. Response: {llm_response}")
                        last_text_response = "(Agent received response with no candidates)"
                        task_completed = True
                        final_summary = last_text_response
                        break

                    response_candidate = llm_response.candidates[0]
                    if not response_candidate.content or not response_candidate.content.parts:
                        log.error(f"LLM response candidate had no content or parts. Candidate: {response_candidate}")
                        last_text_response = "(Agent received response candidate with no content/parts)"
                        task_completed = True
                        final_summary = last_text_response
                        break

                    # --- REVISED LOOP LOGIC FOR MULTI-PART HANDLING ---
                    function_call_part_to_execute = None
                    text_response_buffer = ""
                    processed_function_call_in_turn = (
                        False  # Flag to ensure only one function call is processed per turn
                    )

                    # Iterate through all parts in the response
                    for part in response_candidate.content.parts:
                        if (
                            hasattr(part, "function_call")
                            and part.function_call
                            and not processed_function_call_in_turn
                        ):
                            function_call = part.function_call
                            tool_name = function_call.name
                            tool_args = dict(function_call.args) if function_call.args else {}
                            log.info(f"LLM requested Function Call: {tool_name} with args: {tool_args}")

                            # Add the function *call* part to history immediately
                            self.add_to_history({"role": "model", "parts": [part]})
                            self._manage_context_window()

                            # Store details for execution after processing all parts
                            function_call_part_to_execute = part
                            processed_function_call_in_turn = (
                                True  # Mark that we found and will process a function call
                            )
                            # Don't break here yet, process other parts (like text) first for history/logging

                        elif hasattr(part, "text") and part.text:
                            llm_text = part.text
                            log.info(f"LLM returned text part (Iter {iteration_count}): {llm_text[:100]}...")
                            text_response_buffer += llm_text + "\n"  # Append text parts
                            # Add the text response part to history
                            self.add_to_history({"role": "model", "parts": [part]})
                            self._manage_context_window()

                        else:
                            log.warning(f"LLM returned unexpected response part (Iter {iteration_count}): {part}")
                            # Add it to history anyway?
                            self.add_to_history({"role": "model", "parts": [part]})
                            self._manage_context_window()

                    # --- Now, decide action based on processed parts ---
                    if function_call_part_to_execute:
                        # === Execute the Tool === (Using stored details)
                        function_call = function_call_part_to_execute.function_call  # Get the stored call
                        tool_name = function_call.name
                        tool_args = dict(function_call.args) if function_call.args else {}

                        tool_result = ""
                        tool_error = False
                        user_rejected = False  # Flag for user rejection

                        # --- HUMAN IN THE LOOP CONFIRMATION ---
                        if tool_name in ["edit", "create_file"]:
                            file_path = tool_args.get("file_path", "(unknown file)")
                            content = tool_args.get("content")  # Get content, might be None
                            old_string = tool_args.get("old_string")  # Get old_string
                            new_string = tool_args.get("new_string")  # Get new_string

                            panel_content = f"[bold yellow]Proposed Action:[/bold yellow]\n[cyan]Tool:[/cyan] {tool_name}\n[cyan]File:[/cyan] {file_path}\n"

                            if content is not None:  # Case 1: Full content provided
                                # Prepare content preview (limit length?)
                                preview_lines = content.splitlines()
                                max_preview_lines = 30  # Limit preview for long content
                                if len(preview_lines) > max_preview_lines:
                                    content_preview = (
                                        "\n".join(preview_lines[:max_preview_lines])
                                        + f"\n... ({len(preview_lines) - max_preview_lines} more lines)"
                                    )
                                else:
                                    content_preview = content
                                panel_content += f"\n[bold]Content Preview:[/bold]\n---\n{content_preview}\n---"

                            elif old_string is not None and new_string is not None:  # Case 2: Replacement
                                max_snippet = 50  # Max chars to show for old/new strings
                                old_snippet = old_string[:max_snippet] + (
                                    "..." if len(old_string) > max_snippet else ""
                                )
                                new_snippet = new_string[:max_snippet] + (
                                    "..." if len(new_string) > max_snippet else ""
                                )
                                panel_content += f"\n[bold]Action:[/bold] Replace occurrence of:\n---\n{old_snippet}\n---\n[bold]With:[/bold]\n---\n{new_snippet}\n---"
                            else:  # Case 3: Other/Unknown edit args
                                panel_content += "\n[italic](Preview not available for this edit type)"

                            action_desc = (
                                f"Change: {old_string} to {new_string}"
                                if old_string and new_string
                                else "(No change specified)"
                            )
                            panel_content += f"\n[cyan]Change:[/cyan]\n{action_desc}"

                            # Use full path for Panel
                            self.console.print(
                                rich.panel.Panel(
                                    panel_content,
                                    title="Confirmation Required",
                                    border_style="red",
                                    expand=False,
                                )
                            )

                            # Use questionary for confirmation
                            confirmed = questionary.confirm(
                                "Apply this change?",
                                default=False,  # Default to No
                                auto_enter=False,  # Require Enter key press
                            ).ask()

                            # Handle case where user might Ctrl+C during prompt
                            if confirmed is None:
                                log.warning("User cancelled confirmation prompt.")
                                tool_result = f"User cancelled confirmation for {tool_name} on {file_path}."
                                user_rejected = True
                            elif not confirmed:  # User explicitly selected No
                                log.warning(f"User rejected proposed action: {tool_name} on {file_path}")
                                tool_result = f"User rejected the proposed {tool_name} operation on {file_path}."
                                user_rejected = True  # Set flag to skip execution
                            else:  # User selected Yes
                                log.info(f"User confirmed action: {tool_name} on {file_path}")
                        # --- END CONFIRMATION ---

                        # Only execute if not rejected by user
                        if not user_rejected:
                            status_msg = f"Executing {tool_name}"
                            if tool_args:
                                status_msg += f" ({', '.join([f'{k}={str(v)[:30]}...' if len(str(v)) > 30 else f'{k}={v}' for k, v in tool_args.items()])})"

                            with self.console.status(f"[yellow]{status_msg}...", spinner="dots"):
                                try:
                                    tool_instance = get_tool(tool_name)
                                    if tool_instance:
                                        log.debug(f"Executing tool '{tool_name}' with arguments: {tool_args}")
                                        tool_result = tool_instance.execute(**tool_args)
                                        log.info(
                                            f"Tool '{tool_name}' executed. Result length: {len(str(tool_result)) if tool_result else 0}"
                                        )
                                        log.debug(f"Tool '{tool_name}' result: {str(tool_result)[:500]}...")
                                    else:
                                        log.error(f"Tool '{tool_name}' not found.")
                                        tool_result = f"Error: Tool '{tool_name}' is not available."
                                        tool_error = True
                                except Exception as tool_exec_error:
                                    log.error(
                                        f"Error executing tool '{tool_name}' with args {tool_args}: {tool_exec_error}",
                                        exc_info=True,
                                    )
                                    tool_result = f"Error executing tool {tool_name}: {str(tool_exec_error)}"
                                    tool_error = True

                                # --- Print Executed/Error INSIDE the status block ---
                                if tool_error:
                                    self.console.print(
                                        f"[red] -> Error executing {tool_name}: {str(tool_result)[:100]}...[/red]"
                                    )
                                else:
                                    self.console.print(f"[dim] -> Executed {tool_name}[/dim]")
                            # --- End Status Block ---

                        # === Check for Task Completion Signal via Tool Call ===
                        if tool_name == "task_complete":
                            log.info("Task completion signaled by 'task_complete' function call.")
                            task_completed = True
                            final_summary = tool_result  # The result of task_complete IS the summary
                            # We break *after* adding the function response below

                        # === Add Function Response to History ===
                        # Create the FunctionResponse proto
                        function_response_proto = genai.FunctionResponse(
                            name=tool_name,
                            response={"result": tool_result},  # API expects dict
                        )
                        # Wrap it in a Part proto
                        response_part_proto = genai.Part(function_response=function_response_proto)

                        # Append to history
                        self.add_to_history(
                            {
                                "role": "user",  # Function response acts as a 'user' turn providing data
                                "parts": [response_part_proto],
                            }
                        )
                        self._manage_context_window()

                        if task_completed:
                            break  # Exit loop NOW that task_complete result is in history
                        else:
                            continue  # IMPORTANT: Continue loop to let LLM react to function result

                    elif text_response_buffer:
                        # === Only Text Returned ===
                        log.info(
                            "LLM returned only text response(s). Assuming task completion or explanation provided."
                        )
                        last_text_response = text_response_buffer.strip()
                        task_completed = True  # Treat text response as completion
                        final_summary = last_text_response  # Use the text as the summary
                        break  # Exit the loop

                    else:
                        # === No actionable parts found ===
                        log.warning("LLM response contained no actionable parts (text or function call).")
                        last_text_response = "(Agent received response with no actionable parts)"
                        task_completed = True  # Treat as completion to avoid loop errors
                        final_summary = last_text_response
                        break  # Exit loop

                except google.api_core.exceptions.ResourceExhausted as quota_error:
                    log.warning(f"Quota exceeded for model '{self.current_model_name}': {quota_error}")
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
                        self.console.print(
                            f"[bold yellow]Quota limit reached for {self.current_model_name}. Switching to fallback model ({FALLBACK_MODEL})...[/bold yellow]"
                        )
                        self.current_model_name = FALLBACK_MODEL
                        try:
                            self._initialize_model_instance()  # Recreate model instance with fallback name
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
                    # Clean history
                    if self.history[-1]["role"] == "user":
                        self.history.pop()
                    return f"Error during agent processing: {generation_error}"

            # === End Agent Loop ===

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
            return f"An unexpected error occurred during the agent process: {str(e)}"

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
    def _create_tool_definitions(self) -> list[genai_types.Tool] | None:
        """Dynamically create Tool definitions from AVAILABLE_TOOLS."""
        # NOTE: This assumes get_function_declaration() returns objects compatible with or convertible to genai_types.Tool
        declarations = []
        for tool_name, tool_instance in AVAILABLE_TOOLS.items():
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
        """Finds the last text part sent by the model in the history."""
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "model":
                try:
                    # Check if parts exists and has content
                    if history[i]["parts"] and hasattr(history[i]["parts"][0], "text"):
                        return history[i]["parts"][0].text.strip()
                except (AttributeError, IndexError):
                    continue  # Ignore malformed history entries
        return "(No previous model text found)"

    # --- Add Gemini-specific history management methods ---
    def add_to_history(self, entry):
        """Adds an entry to the Gemini conversation history."""
        self.history.append(entry)
        self._manage_context_window()  # Call truncation logic after adding

    def clear_history(self):
        """Clears the Gemini conversation history, preserving the system prompt."""
        if self.history:
            self.history = self.history[:2]  # Keep user(system_prompt) and initial model response
        else:
            self.history = []  # Should not happen if initialized correctly
        log.info("Gemini history cleared.")
