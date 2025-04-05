"""
Main entry point for the CLI Code Agent application.
Targets Gemini 2.5 Pro Experimental. Includes ASCII Art welcome.
Passes console object to model.
"""

import logging
import os
import sys
import time

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import Config

# Remove list_available_models import:
# from .models.gemini import GeminiModel, list_available_models
from .models.base import AbstractModelAgent  # Keep base import

# Import the specific model classes (adjust path if needed)
# We will dynamically import/instantiate later based on provider
from .models.gemini import GeminiModel  # Keep GeminiModel import
from .models.ollama import OllamaModel  # Import the new Ollama agent
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

    agent_instance: AbstractModelAgent | None = None
    models_list: list[dict] | None = None

    try:
        # --- Instantiate the correct agent ---
        if target_provider == "gemini":
            agent_instance = GeminiModel(api_key=credential, console=console, model_name=None)
        elif target_provider == "ollama":
            # Instantiate OllamaModel
            agent_instance = OllamaModel(api_url=credential, console=console, model_name=None)
        else:
            console.print(f"[bold red]Error:[/bold red] Unknown provider '{target_provider}'.")
            return

        # --- Call the agent's list_models method ---
        models_list = agent_instance.list_models()

        # --- Process and display results ---
        if models_list is None:
            # Error message should have been printed by the agent's list_models method
            log.warning(f"Agent's list_models returned None for provider {target_provider}.")
            # console.print(f"[red]Failed to list models for {target_provider}. Check logs.")
            return  # Exit if listing failed

        if not models_list:
            console.print(f"[yellow]No models found or reported by provider '{target_provider}'.[/yellow]")
            return

        console.print(f"\n[bold cyan]Available {target_provider.capitalize()} Models:[/bold cyan]")
        for model_data in models_list:
            # Assuming model_data is a dict with at least 'id' and 'name'
            model_id = model_data.get("id", "N/A")
            display_name = model_data.get("name", model_id)  # Use name, fallback to id
            console.print(f"- [bold green]{model_id}[/bold green] (Name: {display_name})")

        # Display current default for this provider
        current_default = config.get_default_model(provider=target_provider)
        if current_default:
            console.print(f"\nCurrent default {target_provider.capitalize()} model: {current_default}")
        else:
            console.print(f"\nNo default model set for {target_provider.capitalize()}.")

        console.print(
            f"\nUse 'cli-code --provider={target_provider} --model MODEL' or 'cli-code set-default-model --provider={target_provider} MODEL'."
        )

    except Exception as e:
        console.print(f"[bold red]Error listing models for {target_provider}:[/bold red] {e}")
        log.error(f"List models command failed for {target_provider}", exc_info=True)


# --- MODIFIED start_interactive_session ---
def start_interactive_session(provider: str, model_name: str, console: Console):
    """Start an interactive chat session with the selected provider and model."""
    if not config:
        console.print("[bold red]Config error.[/bold red]")
        return

    # --- Display Welcome Art ---
    console.clear()
    console.print(CLI_CODE_ART)  # Use updated art name
    console.print(
        Panel(
            f"[b]Welcome to CLI Code AI Assistant! (Provider: {provider.capitalize()})[/b]",
            border_style="blue",
            expand=False,
        )
    )
    time.sleep(0.1)
    # --- End Welcome Art ---

    credential = config.get_credential(provider)
    # Check if credential exists and log its source (env var or config file)
    if credential:
        cred_type = "API Key" if provider == "gemini" else "API URL"
        env_var = "CLI_CODE_GOOGLE_API_KEY" if provider == "gemini" else "CLI_CODE_OLLAMA_API_URL"
        if env_var in os.environ:
            log.info(f"Using {provider} {cred_type} from environment variable {env_var}")
        else:
            log.info(f"Using {provider} {cred_type} from config file")
    else:
        credential_type = "API Key" if provider == "gemini" else "API URL"
        console.print(f"\n[bold red]Error:[/bold red] {provider.capitalize()} {credential_type} not found.")
        console.print(
            f"Please run [bold]'cli-code setup --provider={provider} YOUR_{credential_type.upper().replace(' ', '_')}'[/bold] first."
        )
        console.print(f"Or set the environment variable [bold]CLI_CODE_{provider.upper()}_API_{'KEY' if provider == 'gemini' else 'URL'}[/bold]")
        return

    try:
        console.print(f"\nInitializing provider [bold]{provider}[/bold] with model [bold]{model_name}[/bold]...")

        model_agent: AbstractModelAgent | None = None  # Define agent variable

        # --- Instantiate the correct agent ---
        if provider == "gemini":
            model_agent = GeminiModel(api_key=credential, console=console, model_name=model_name)
            console.print("[green]Gemini model initialized successfully.[/green]")
        elif provider == "ollama":
            # Instantiate OllamaModel agent
            model_agent = OllamaModel(api_url=credential, console=console, model_name=model_name)
            console.print("[green]Ollama provider initialized successfully.[/green]")
        else:
            console.print(f"[bold red]Error:[/bold red] Unknown provider '{provider}'. Cannot initialize.")
            log.error(f"Attempted to start session with unknown provider: {provider}")
            return
        
        # Add information about context initialization (for all successful provider initializations)
        if os.path.isdir(".rules"):
            md_files = [f for f in os.listdir(".rules") if f.endswith(".md")]
            file_count = len(md_files)
            if file_count > 0:
                file_str = "file" if file_count == 1 else "files"
                console.print(f"[dim]Context will be initialized from {file_count} .rules/*.md {file_str}.[/dim]")
            else:
                console.print("[dim]Context will be initialized from directory listing (ls) - .rules directory exists but contains no .md files.[/dim]")
        elif os.path.isfile("README.md"):
            console.print("[dim]Context will be initialized from README.md.[/dim]")
        else:
            console.print("[dim]Context will be initialized from directory listing (ls).[/dim]")
        console.print()  # Empty line for spacing

    except Exception as e:
        console.print(f"\n[bold red]Error initializing model '{model_name}':[/bold red] {e}")
        log.error(f"Failed to initialize model {model_name}", exc_info=True)
        console.print("Please check model name, API key permissions, network. Use 'cli-code list-models'.")
        return

    # --- Session Start Message ---
    console.print("Type '/help' for commands, '/exit' or Ctrl+C to quit.")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")

            if user_input.lower() == "/exit":
                break
            elif user_input.lower() == "/help":
                show_help(provider)
                continue  # Pass provider to help

            # Display initial "thinking" status - generate handles intermediate ones
            response_text = model_agent.generate(user_input)  # Use the instantiated agent

            if response_text is None and user_input.startswith("/"):
                console.print(f"[yellow]Unknown command:[/yellow] {user_input}")
                continue
            elif response_text is None:
                console.print("[red]Received an empty response from the model.[/red]")
                log.warning("generate() returned None unexpectedly.")
                continue

            # --- Changed Prompt Name ---
            console.print("[bold medium_purple]Assistant:[/bold medium_purple]")  # Changed from provider.capitalize()
            console.print(Markdown(response_text), highlight=True)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]An error occurred during the session:[/bold red] {e}")
            log.error("Error during interactive loop", exc_info=True)


def show_help(provider: str):
    """Show available commands for the interactive mode."""
    help_text = f"""
    [bold blue]CLI Code Agent Help[/bold blue]
    
    [bold yellow]Chat Commands:[/bold yellow]
      /exit    - Exit the chat session
      /help    - Show this help message
      
    [bold yellow]Configuration Commands:[/bold yellow] (in terminal)
      cli-code-agent setup --provider={provider} <api_key_or_url>     - Configure API key/URL
      cli-code-agent list-models --provider={provider}                - List available models
      cli-code-agent set-default-model --provider={provider} <model>  - Set default model
      
    [bold yellow]Tips:[/bold yellow]
      - Ask about files or code in your current directory
      - Request file operations (view/edit/create) 
      - Ask for explanations of code
      - Create or modify projects with the AI's help
      - Run commands (with your approval)
    """
    return Markdown(help_text)


if __name__ == "__main__":
    cli(obj={})
