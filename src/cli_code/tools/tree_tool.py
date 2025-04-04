"""
Tool for displaying directory structure using the 'tree' command.
"""
import subprocess
import logging
from google.generativeai.types import FunctionDeclaration, Tool

from .base import BaseTool

log = logging.getLogger(__name__)

DEFAULT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 10

class TreeTool(BaseTool):
    name: str = "tree"
    description: str = (
        f"""Displays the directory structure as a tree. Shows directories and files.
        Use this to understand the hierarchy and layout of the current working directory or a subdirectory.
        Defaults to a depth of {DEFAULT_TREE_DEPTH}. Use the 'depth' argument to specify a different level.
        Optionally specify a 'path' to view a subdirectory instead of the current directory."""
    )
    args_schema: dict = {
         "path": {
            "type": "string",
            "description": "Optional path to a specific directory relative to the workspace root. If omitted, uses the current directory.",
        },
        "depth": {
            "type": "integer",
            "description": f"Optional maximum display depth of the directory tree (Default: {DEFAULT_TREE_DEPTH}, Max: {MAX_TREE_DEPTH}).",
        },
    }
    # Optional args: path, depth
    required_args: list[str] = []

    def execute(self, path: str | None = None, depth: int | None = None) -> str:
        """Executes the tree command."""
        
        if depth is None:
            depth_limit = DEFAULT_TREE_DEPTH
        else:
            # Clamp depth to be within reasonable limits
            depth_limit = max(1, min(depth, MAX_TREE_DEPTH))
            
        command = ['tree', f'-L {depth_limit}']
        
        # Add path if specified
        target_path = "." # Default to current directory
        if path:
            # Basic path validation/sanitization might be needed depending on security context
            target_path = path
            command.append(target_path)

        log.info(f"Executing tree command: {' '.join(command)}")
        try:
            # Adding '-a' might be useful to show hidden files, but could be verbose.
            # Adding '-F' appends / to dirs, * to executables, etc.
            # Using shell=True is generally discouraged, but might be needed if tree isn't directly in PATH
            # or if handling complex paths. Sticking to list format for now.
            process = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=False, # Don't raise exception on non-zero exit code
                timeout=15 # Add a timeout
            )

            if process.returncode == 0:
                log.info(f"Tree command successful for path '{target_path}' with depth {depth_limit}.")
                # Limit output size? Tree can be huge.
                output = process.stdout.strip()
                if len(output.splitlines()) > 200: # Limit lines as a proxy for size
                     log.warning(f"Tree output for '{target_path}' exceeded 200 lines. Truncating.")
                     output = "\n".join(output.splitlines()[:200]) + "\n... (output truncated)"
                return output
            elif process.returncode == 127 or "command not found" in process.stderr.lower():
                 log.error(f"\'tree\' command not found. It might not be installed.")
                 return "Error: 'tree' command not found. Please ensure it is installed and in the system's PATH."
            else:
                log.error(f"Tree command failed with return code {process.returncode}. Path: '{target_path}', Depth: {depth_limit}. Stderr: {process.stderr.strip()}")
                error_detail = process.stderr.strip() if process.stderr else "(No stderr)"
                return f"Error executing tree command (Code: {process.returncode}): {error_detail}"

        except FileNotFoundError:
             log.error(f"\'tree\' command not found (FileNotFoundError). It might not be installed.")
             return "Error: 'tree' command not found. Please ensure it is installed and in the system's PATH."
        except subprocess.TimeoutExpired:
             log.error(f"Tree command timed out for path '{target_path}' after 15 seconds.")
             return f"Error: Tree command timed out for path '{target_path}'. The directory might be too large or complex."
        except Exception as e:
            log.exception(f"An unexpected error occurred while executing tree command for path '{target_path}': {e}")
            return f"An unexpected error occurred while executing tree: {str(e)}" 