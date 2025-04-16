"""
Main entry point for the CLI Code Agent application.
Targets Gemini 2.5 Pro Experimental. Includes ASCII Art welcome.
Passes console object to model.
"""

import asyncio
import logging
import os
import sys
import time
import uuid
import json

import click
# # Revert to original imports for most, try submodule for Protocol
# from chuk_mcp import (
#     # MCPClientProtocol, # <-- Still not here
#     MCPError,
#     MCPMessage,
#     decode_mcp_message,
#     encode_mcp_message,
# )
# from chuk_mcp.mcp_client.protocol import MCPClientProtocol # <-- Trying this path again
# # Import classes from their likely submodules - Attempt 2
# from chuk_mcp.mcp_client.exceptions import MCPError
# from chuk_mcp.mcp_client.protocol.base_protocol import MCPClientProtocol # Guessing base_protocol
# from chuk_mcp.mcp_client.messages.message_base import MCPMessage # Guessing message_base
# from chuk_mcp.mcp_client.messages.codec import decode_mcp_message, encode_mcp_message # Assuming codec module
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import Config

# Remove list_available_models import:
# from .models.gemini import GeminiModel, list_available_models
from .models.base import AbstractModelAgent  # Keep base import

# Import the specific model classes (adjust path if needed)
# We will dynamically import/instantiate later based on provider
# NO LONGER NEEDED HERE - Remove direct model imports
# from .models.gemini import GeminiModel
# from .models.ollama import OllamaModel
from .tools import AVAILABLE_TOOLS

# Setup console and config
console = Console()  # Create console instance HERE
config = None  # Initialize config as None
try:
    config = Config()
except Exception as e:
    console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
    # Keep config as None if loading failed

# Setup logging - MORE EXPLICIT CONFIGURATION
log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(
    level=log_level, format=log_format, stream=sys.stdout, force=True
)  # Use basicConfig with force=True for simplicity

log = logging.getLogger(__name__)  # Get logger for this module
log.info(f"Logging initialized with level: {log_level}")

# --- Default Model (Provider specific defaults are now in Config) ---
# DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25" # Removed global default

# --- ASCII Art Definition ---
CLI_CODE_ART = r"""

[medium_blue]

 ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░       ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
 ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░       ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░

 [/medium_blue]
"""
# --- End ASCII Art ---


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# --- Provider Choice ---
PROVIDER_CHOICES = click.Choice(["gemini", "ollama"])


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--provider",
    "-p",
    type=PROVIDER_CHOICES,
    default=None,  # Default is determined from config later
    help="Specify the LLM provider to use (e.g., gemini, ollama). Overrides config default.",
)
@click.option(
    "--model",
    "-m",
    default=None,  # Default is determined from config/provider later
    help="Specify the model ID to use. Overrides provider default.",
)
@click.pass_context
def cli(ctx, provider, model):
    """Interactive CLI for the cli-code assistant with coding assistance tools."""
    if not config:
        console.print("[bold red]Configuration could not be loaded. Cannot proceed.[/bold red]")
        sys.exit(1)

    ctx.ensure_object(dict)
    # Store provider and model for subcommands, resolving defaults
    selected_provider = provider or config.get_default_provider()
    selected_model = model  # Keep explicit model if passed

    ctx.obj["PROVIDER"] = selected_provider
    ctx.obj["MODEL"] = selected_model  # Will be None if not passed via CLI

    log.info(
        f"CLI invoked. Determined provider: {selected_provider}, Explicit model: {selected_model or 'Not Specified'}"
    )

    if ctx.invoked_subcommand is None:
        # Resolve model fully if starting interactive session
        final_model = selected_model or config.get_default_model(selected_provider)
        if not final_model:
            console.print(
                f"[bold red]Error:[/bold red] No default model configured for provider '{selected_provider}' and no model specified with --model."
            )
            console.print(
                f"Run 'cli-code set-default-model --provider={selected_provider} YOUR_MODEL_NAME' or use the --model flag."
            )
            sys.exit(1)

        log.info(f"Starting interactive session. Provider: {selected_provider}, Model: {final_model}")
        start_interactive_session(provider=selected_provider, model_name=final_model, console=console)


# --- Refactored Setup Command ---
@cli.command()
@click.option(
    "--provider", "-p", type=PROVIDER_CHOICES, required=True, help="The provider to configure (gemini or ollama)."
)
@click.argument("credential", required=True)
def setup(provider, credential):
    """Configure credentials (API Key/URL) for a specific provider."""
    if not config:
        console.print("[bold red]Config error.[/bold red]")
        return

    credential_type = "API Key" if provider == "gemini" else "API URL"

    try:
        config.set_credential(provider, credential)
        # Also set as default provider on first successful setup for that provider? Optional.
        # config.set_default_provider(provider)
        console.print(f"[green]✓[/green] {provider.capitalize()} {credential_type} saved.")
        if provider == "ollama":
            console.print(f"[yellow]Note:[/yellow] Ensure your Ollama server is running and accessible at {credential}")
            console.print(
                "You may need to set a default model using 'cli-code set-default-model --provider=ollama MODEL_NAME'."
            )
        elif provider == "gemini":
            console.print(f"Default model is currently set to: {config.get_default_model(provider='gemini')}")

    except Exception as e:
        console.print(f"[bold red]Error saving {credential_type}:[/bold red] {e}")
        log.error(f"Failed to save credential for {provider}", exc_info=True)


# --- New Set Default Provider Command ---
@cli.command()
@click.argument("provider", type=PROVIDER_CHOICES, required=True)
def set_default_provider(provider):
    """Set the default LLM provider to use."""
    if not config:
        console.print("[bold red]Config error.[/bold red]")
        return
    try:
        config.set_default_provider(provider)
        console.print(f"[green]✓[/green] Default provider set to [bold]{provider}[/bold].")
    except Exception as e:
        console.print(f"[bold red]Error setting default provider:[/bold red] {e}")
        log.error(f"Failed to set default provider to {provider}", exc_info=True)


# --- Refactored Set Default Model Command ---
@cli.command()
@click.option(
    "--provider",
    "-p",
    type=PROVIDER_CHOICES,
    default=None,  # If None, uses the current default provider
    help="Set the default model for this specific provider.",
)
@click.argument("model_name", required=True)
@click.pass_context  # Need context to get the default provider if --provider is not used
def set_default_model(ctx, provider, model_name):
    """Set the default model ID for a provider."""
    if not config:
        console.print("[bold red]Config error.[/bold red]")
        return

    target_provider = provider or config.get_default_provider()  # Use flag or config default

    try:
        config.set_default_model(model_name, provider=target_provider)
        console.print(
            f"[green]✓[/green] Default model for provider [bold]{target_provider}[/bold] set to [bold]{model_name}[/bold]."
        )
    except Exception as e:
        console.print(f"[bold red]Error setting default model for {target_provider}:[/bold red] {e}")
        log.error(f"Failed to set default model {model_name} for {target_provider}", exc_info=True)


# --- Refactored List Models Command ---
@cli.command()
@click.option(
    "--provider",
    "-p",
    type=PROVIDER_CHOICES,
    default=None,  # If None, uses the current default provider
    help="List models available for a specific provider.",
)
def list_models(provider):
    """List available models for a configured provider."""
    if not config:
        console.print("[bold red]Config error.[/bold red]")
        return

    target_provider = provider or config.get_default_provider()
    credential = config.get_credential(target_provider)

    if not credential:
        credential_type = "API Key" if target_provider == "gemini" else "API URL"
        console.print(f"[bold red]Error:[/bold red] {target_provider.capitalize()} {credential_type} not found.")
        console.print(
            f"Please run 'cli-code setup --provider={target_provider} YOUR_{credential_type.upper().replace(' ', '_')}' first."
        )
        return

    console.print(f"[yellow]Fetching models for provider '{target_provider}'...[/yellow]")

    # --- REFACTOR MCP: Client cannot directly list models anymore --- #
    # agent_instance: AbstractModelAgent | None = None
    # models_list: list[dict] | None = None

    # try:
    #     # --- Instantiate the correct agent ---
    #     if target_provider == "gemini":
    #         # F821 Error Here - commenting out
    #         # agent_instance = GeminiModel(api_key=credential, console=console, model_name=None)
    #         pass
    #     elif target_provider == "ollama":
    #         # Instantiate OllamaModel
    #         # F821 Error Here - commenting out
    #         # agent_instance = OllamaModel(api_url=credential, console=console, model_name=None)
    #         pass
    #     else:
    #         console.print(f"[bold red]Error:[/bold red] Unknown provider '{target_provider}'.")
    #         return
    #
    #     # --- Call the agent's list_models method ---
    #     models_list = agent_instance.list_models()
    #
    #     # --- Process and display results ---
    #     if models_list is None:
    #         log.warning(f"Agent's list_models returned None for provider {target_provider}.")
    #
    #     elif not models_list: # Handle empty list explicitly
    #         console.print(f"[yellow]No models found for {target_provider}.[/yellow]")
    #
    #     else:
    #         console.print(f"[bold]Available models for {target_provider}:[/bold]")
    #         for model_info in models_list:
    #             model_id = model_info.get("id", "N/A")
    #             model_name = model_info.get("name", "Unknown") # Use name if available
    #             console.print(f"  - {model_id} ({model_name})")
    #
    # except Exception as e:
    #     console.print(f"\n[bold red]Error listing models for {target_provider}:[/bold red] {e}")
    #     log.error(f"List models command failed for {target_provider}", exc_info=True)

    console.print("[bold yellow]MCP Refactor:[/bold yellow] Model listing must now be requested from the MCP server.")
    console.print("(Functionality temporarily disabled in client)")


# --- MCP Configuration
MCP_SERVER_HOST = "127.0.0.1"
MCP_SERVER_PORT = 8999
# --- End MCP Configuration ---


# --- MODIFIED start_interactive_session ---
def start_interactive_session(provider: str, model_name: str, console: Console):
    """Starts an interactive chat session using the MCP protocol."""
    console.print(Panel(CLI_CODE_ART, border_style="medium_blue", title="cli-code"))
    console.print(
        f"[dim]Provider: '{provider}', Model: '{model_name}' (Note: MCP Stub Server ignores these for now)[/dim]"
    )
    console.print("[dim]Type '/exit' to quit, '/help' for commands.[/dim]")

    agent_id = f"cli-code-client-{uuid.uuid4()!s:.8}"
    session_id = f"session-{uuid.uuid4()!s:.8}"

    log.info(f"Starting MCP interactive session. AgentID: {agent_id}, SessionID: {session_id}")

    # Run the async session handler
    try:
        asyncio.run(run_mcp_session(agent_id, session_id, console))
    except ConnectionRefusedError:
        console.print(
            f"[bold red]Connection Error:[/bold red] Could not connect to MCP Server at {MCP_SERVER_HOST}:{MCP_SERVER_PORT}."
        )
        console.print("[yellow]Is the stub server (mcp_stub_server.py) running?[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        log.error("Unexpected error in MCP session", exc_info=True)


# --- NEW ASYNC FUNCTION ---
async def run_mcp_session(agent_id: str, session_id: str, console: Console):
    """Handles the asynchronous MCP client connection and interaction loop."""

    reader, writer = await asyncio.open_connection(MCP_SERVER_HOST, MCP_SERVER_PORT)
    # mcp_protocol = MCPClientProtocol(reader, writer) # Removed protocol instance
    log.info(f"Connected to MCP Server at {MCP_SERVER_HOST}:{MCP_SERVER_PORT}")

    try:
        while True:
            # Initialize message_type and text_payload for this turn
            message_type = None
            text_payload = None
            try:
                # Correctly await the result of console.input run in a thread
                user_input_str = await asyncio.to_thread(console.input, "[bold cyan]You:[/bold cyan] ")
                user_input = user_input_str.strip()

                if not user_input:
                    continue

                if user_input.lower() == "/exit":
                    console.print("[yellow]Exiting session.[/yellow]")
                    break
                elif user_input.lower() == "/help":
                    # TODO: Implement more detailed help if needed
                    console.print("[bold]Available commands:[/bold]")
                    console.print("  /exit - Quit the session")
                    console.print("  /help - Show this help message")
                    continue

                # Send user message via MCP
                # Construct the message payload manually
                user_payload = {
                    "message_type": "user_message",
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "payload": {"text": user_input},
                }
                # Encode and send
                encoded_message = json.dumps(user_payload).encode('utf-8') + b'\n'
                writer.write(encoded_message)
                await writer.drain()
                # await mcp_protocol.send_message(user_msg) # Removed old send
                log.info(f"Sent User Message: {user_input[:50]}...")

                # --- Receive and process server responses --- M1: Basic Text Only ---
                # Move status outside the inner processing loop if it causes issues
                # with console.status("[dim]Waiting for response...[/dim]", spinner="dots"):
                while True:  # Loop to potentially handle multiple messages (e.g., status, then final)
                    try:
                        # Read line by line
                        line = await reader.readline()
                        if not line:
                            console.print("[bold red]Connection closed by server.[/bold red]")
                            return # Exit session

                        # Decode JSON
                        try:
                            response_data = json.loads(line.decode('utf-8'))
                        except json.JSONDecodeError as json_err:
                            log.error(f"Failed to decode JSON from server: {line.decode('utf-8', errors='ignore')}", exc_info=json_err)
                            console.print("[bold red]Error:[/bold red] Received invalid data from server.")
                            break # Exit inner receive loop on decode error

                        log.info(f"Received MCP Data: {response_data}") # Log raw data

                        # Process based on type
                        message_type = response_data.get("message_type")
                        payload = response_data.get("payload", {})

                        if message_type == "assistant_message":
                            text_payload = payload.get("text", "")
                            if text_payload:
                                log.debug(f"Printing assistant message payload: {text_payload!r}")
                                # Print message *after* potential status spinner is implicitly exited by break
                                # console.print(Markdown(text_payload))
                            # For M1, assume one assistant message ends the turn
                            break  # Exit receive loop, wait for next user input
                        elif message_type == "error_message":
                            error_payload = payload.get("message", "Unknown error from server.")
                            console.print(f"[bold red]Server Error:[/bold red] {error_payload}")
                            break  # Exit receive loop
                        elif message_type == "status_update":  # Example future handling
                            status_payload = payload.get("status", "Server is working...")
                            # Display status update within the loop
                            with console.status(f"[dim]{status_payload}...[/dim]", spinner="dots"):
                                await asyncio.sleep(0.1) # Keep status visible briefly
                            # console.print(f"[dim]Status:[/dim] {status_payload}") # Old way
                            # Continue waiting for the final message in this turn
                        else:
                            # Handle other message types (tool calls, etc.) in later milestones
                            console.print(
                                f"[dim]Received unhandled message type: {message_type}[/dim]"
                            )
                            break  # For M1, break on anything unexpected

                    except asyncio.TimeoutError:
                        console.print("[yellow]Timeout waiting for server response. Try again?[/yellow]")
                        break  # Exit receive loop, let user retry
                    except Exception as e:
                        console.print(f"[bold red]Error processing server message:[/bold red] {e}")
                        log.error("Error receiving/processing message", exc_info=True)
                        break  # Exit receive loop

                # --- Print final message outside the receive loop ---
                if message_type == "assistant_message" and text_payload:
                    console.print(Markdown(text_payload))

            except EOFError:
                console.print("[yellow]Input stream closed. Exiting.[/yellow]")
                break
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
                continue  # Allow user to type /exit

    except Exception as e:
        console.print(f"[bold red]Session Error:[/bold red] {e}")
        log.error("Error in MCP session outer loop", exc_info=True)
    finally:
        log.info("Closing MCP connection.")
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    # Provide default None for linter satisfaction, Click handles actual values
    cli(ctx=None, provider=None, model=None, obj={})
