# chuk_mcp/chuk_mcp.mcp_client/host/server_manager.py
import asyncio
import json
import logging
import os
from asyncio.streams import StreamReader, StreamWriter
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import anyio
from anyio.abc import TaskGroup

# cli imports
from mcp_code.config import load_config
from mcp_code.mcp_client.messages.initialize.send_messages import send_initialize

# mcp imports
from mcp_code.mcp_client.transport.stdio.stdio_client import stdio_client
from mcp_code.mcp_client.transport.stdio.stdio_server_parameters import StdioServerParameters

# Timeout for server cleanup in seconds
CLEANUP_TIMEOUT = 5.0

# Default config path
CONFIG_PATH = os.path.expanduser("~/.mcp/config.json")

from mcp_code.mcp_client.language_client import LanguageClient


async def _connect_and_execute(
    command_func: Callable[[Any], Awaitable[Dict[str, Any]]],
    server_params: Dict[str, Any],
    server_name: str,
    results: Dict[str, Any],
    silent: bool = False,
):
    """
    Connect to a server and execute a command, storing the result.

    Args:
        command_func: The async function to execute.
        server_params: The StdioServerParameters for the server.
        server_name: The name of the server.
        results: Dictionary to store the results in.
        silent: Whether to suppress error messages.
    """
    try:
        # Get stdio client context manager
        client_cm = stdio_client(server_params)
        # Connect to the server
        async with client_cm as streams:
            # Initialize the server
            init_result = await send_initialize(streams[0], streams[1], server_params)
            if not init_result:
                if not silent:
                    print(f"Initialization failed for server: {server_name}")
                return

            # Create server info for compatibility with different command_func signatures
            server_info = [{"name": server_name, "streams": streams, "user_specified": True}]

            # Call command_func with the expected signature for the test
            result = await command_func([streams], server_info=server_info)

            # Store the result
            results[server_name] = result
    except Exception as e:
        if not silent:
            print(f"\nError executing command on server {server_name}: {str(e)}")


async def run_command(
    command_func: Callable[[Any], Awaitable[Dict[str, Any]]],
    server: str = None,
    all_servers: bool = False,
    silent: bool = False,
    config_path: str = CONFIG_PATH,
    task_group: TaskGroup = None,
    config_file: str = None,
    server_names: list = None,
    user_specified_servers: bool = False,
) -> Dict[str, Any]:
    """
    Execute the specified command function against one or more language servers.

    Args:
        command_func: The async function to execute (takes a LanguageClient and returns a dict).
        server: Optional name of a specific server to target.
        all_servers: If True, execute against all configured servers.
        silent: If True, don't print connection errors.
        config_path: Path to the configuration file (default: ~/.mcp/config.json).
        task_group: An optional task group to use for concurrent execution.
        config_file: (Deprecated) Use config_path instead.
        server_names: (Deprecated) Use server or all_servers instead.
        user_specified_servers: (Deprecated) No longer used directly.

    Returns:
        A dictionary of server names to results, or a single result if server parameter specified.
    """
    # Handle deprecated parameters (for backward compatibility if needed, else remove)
    if config_file is not None:
        config_path = config_file

    # Load the entire configuration ONCE
    try:
        config = await load_config(config_path)
        mcp_servers_config = config.get("mcpServers", {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if not silent:
            print(f"\nError loading configuration from {config_path}: {str(e)}")
        return {}
    except Exception as e:
        if not silent:
            print(f"\nUnexpected error loading configuration from {config_path}: {str(e)}")
        logging.exception(f"Unexpected error loading config: {config_path}")
        return {}

    # Determine which servers to use
    server_name_list = []
    if server:
        if server in mcp_servers_config:
            server_name_list = [server]
        else:
            if not silent:
                print(f"\nServer '{server}' not found in configuration: {config_path}")
            return {}
    elif all_servers:
        server_name_list = list(mcp_servers_config.keys())
    else:
        # Use the default server if configured
        default_server = config.get("defaultServer")
        if default_server and default_server in mcp_servers_config:
            server_name_list = [default_server]
        elif mcp_servers_config:  # Fallback to first server if no default
            server_name_list = [list(mcp_servers_config.keys())[0]]

    if not server_name_list:
        if not silent:
            print(f"\nNo target servers found or configured in: {config_path}")
        return {}

    # Initialize result dictionary
    results = {}
    own_task_group = False

    try:
        # Create a task group if none was provided
        if task_group is None:
            task_group = anyio.create_task_group()
            own_task_group = True
            await task_group.__aenter__()

        # Connect to each server and execute the command
        for server_name in server_name_list:
            try:
                # Extract the specific server configuration from the already loaded config
                server_config = mcp_servers_config.get(server_name)

                if not server_config:
                    # This shouldn't happen if server_name_list logic is correct, but check anyway
                    if not silent:
                        print(f"\nInternal Error: Server '{server_name}' config missing after initial load.")
                    continue  # Skip this server

                # Construct StdioServerParameters here
                try:
                    server_params = StdioServerParameters(
                        command=server_config["command"],  # Required
                        args=server_config.get("args", []),
                        env=server_config.get("env"),
                        name=server_name,
                    )
                except KeyError as e:
                    if not silent:
                        print(f"\nConfiguration error for server '{server_name}': Missing required key '{e}'")
                    continue  # Skip this misconfigured server

                # Create a task for this server connection using start_soon
                task_group.start_soon(
                    _connect_and_execute,
                    command_func,
                    server_params,  # Pass the constructed StdioServerParameters
                    server_name,
                    results,
                    silent,
                )
            except Exception as e:
                # Handle unexpected errors during task setup for a specific server
                if not silent:
                    print(f"\nError setting up task for server {server_name}: {str(e)}")
                logging.exception(f"Error setting up task for {server_name}")

        # If we own the task group, wait for all tasks to complete
        if own_task_group:
            await task_group.__aexit__(None, None, None)

    except (KeyboardInterrupt, anyio.get_cancelled_exc_class()) as e:
        # Handle keyboard interrupt or task cancellation
        if not silent:
            print("\nCommand interrupted.")

        # If we own the task group, ensure it's properly closed
        if own_task_group:
            try:
                await task_group.__aexit__(type(e), e, None)
            except Exception as aexit_error:
                if not silent:
                    print(f"\nError during cleanup: {aexit_error}")
        # Re-raise KeyboardInterrupt if it was the cause
        if isinstance(e, KeyboardInterrupt):
            raise
    except Exception as e:
        # Handle other exceptions
        if not silent:
            print(f"\nUnexpected error: {str(e)}")

        # If we own the task group, ensure it's properly closed
        if own_task_group:
            try:
                await task_group.__aexit__(type(e), e, None)
            except Exception as aexit_error:
                if not silent:
                    print(f"\nError during cleanup: {aexit_error}")

    # Return appropriately based on input arguments
    if server and isinstance(server, str) and server in results:
        # Add "Command finished." message for successful, non-interactive commands
        # unless explicitly suppressed
        if (
            not silent
            and isinstance(results[server], dict)
            and results[server].get("status") == "success"
            and command_func.__name__ != "interactive_mode"
            and os.environ.get("MCP_AGENT_SUPPRESS_CLEAN_EXIT_LOGS") != "1"
        ):
            print("\nCommand finished.")
        return results[server]
    else:
        # Check for success in any server result for multi-server case
        has_success = any(isinstance(res, dict) and res.get("status") == "success" for res in results.values())
        # Add "Command finished." message for successful, non-interactive commands
        # unless explicitly suppressed
        if (
            not silent
            and has_success
            and command_func.__name__ != "interactive_mode"
            and os.environ.get("MCP_AGENT_SUPPRESS_CLEAN_EXIT_LOGS") != "1"
        ):
            print("\nCommand finished.")
        # Return all results (multiple servers or default)
        return results


# Potential call sites need to be updated to await run_command
# e.g., in src/cli_code/main.py or similar
