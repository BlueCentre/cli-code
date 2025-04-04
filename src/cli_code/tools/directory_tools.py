"""
Tools for directory operations.
"""
import os
import logging
import subprocess
from .base import BaseTool

log = logging.getLogger(__name__)

class CreateDirectoryTool(BaseTool):
    """Tool to create a new directory."""
    name = "create_directory"
    description = "Creates a new directory, including any necessary parent directories."

    def execute(self, dir_path: str) -> str:
        """
        Creates a directory.

        Args:
            dir_path: The path of the directory to create.

        Returns:
            A success or error message.
        """
        try:
            # Basic path safety
            if ".." in dir_path.split(os.path.sep):
                 log.warning(f"Attempted to access parent directory in create_directory path: {dir_path}")
                 return f"Error: Invalid path '{dir_path}'. Cannot access parent directories."

            target_path = os.path.abspath(os.path.expanduser(dir_path))
            log.info(f"Attempting to create directory: {target_path}")

            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    log.warning(f"Directory already exists: {target_path}")
                    return f"Directory already exists: {dir_path}"
                else:
                    log.error(f"Path exists but is not a directory: {target_path}")
                    return f"Error: Path exists but is not a directory: {dir_path}"

            os.makedirs(target_path, exist_ok=True) # exist_ok=True handles race conditions slightly better
            log.info(f"Successfully created directory: {target_path}")
            return f"Successfully created directory: {dir_path}"

        except OSError as e:
            log.error(f"Error creating directory '{dir_path}': {e}", exc_info=True)
            return f"Error creating directory: {str(e)}"
        except Exception as e:
            log.error(f"Unexpected error creating directory '{dir_path}': {e}", exc_info=True)
            return f"Error creating directory: {str(e)}"

class LsTool(BaseTool):
    """Tool to list directory contents using 'ls -lA'."""
    name = "ls"
    description = "Lists the contents of a specified directory (long format, including hidden files)."
    args_schema: dict = {
        "path": {
            "type": "string",
            "description": "Optional path to a specific directory relative to the workspace root. If omitted, uses the current directory.",
        }
    }
    required_args: list[str] = []

    def execute(self, path: str | None = None) -> str:
        """Executes the 'ls -lA' command."""
        target_path = "."  # Default to current directory
        if path:
            # Basic path safety - prevent navigating outside workspace root if needed
            # For simplicity, assuming relative paths are okay for now
            target_path = os.path.normpath(path) # Normalize path
            if target_path.startswith(".."):
                 log.warning(f"Attempted to access parent directory in ls path: {path}")
                 return f"Error: Invalid path '{path}'. Cannot access parent directories."

        command = ['ls', '-lA', target_path]
        log.info(f"Executing ls command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero exit code
                timeout=15    # Add a timeout
            )

            if process.returncode == 0:
                log.info(f"ls command successful for path '{target_path}'.")
                # Limit output size? ls -l can be long.
                output = process.stdout.strip()
                # Example truncation (adjust as needed)
                if len(output.splitlines()) > 100:
                     log.warning(f"ls output for '{target_path}' exceeded 100 lines. Truncating.")
                     output = "\n".join(output.splitlines()[:100]) + "\n... (output truncated)"
                return output
            else:
                # Handle cases like directory not found specifically if possible
                stderr_lower = process.stderr.lower()
                if "no such file or directory" in stderr_lower:
                     log.error(f"ls command failed: Directory not found '{target_path}'. Stderr: {process.stderr.strip()}")
                     return f"Error: Directory not found: '{target_path}'"
                else:
                     log.error(f"ls command failed with return code {process.returncode}. Path: '{target_path}'. Stderr: {process.stderr.strip()}")
                     error_detail = process.stderr.strip() if process.stderr else "(No stderr)"
                     return f"Error executing ls command (Code: {process.returncode}): {error_detail}"

        except FileNotFoundError:
             # This means 'ls' itself wasn't found - unlikely but possible
             log.error("'ls' command not found (FileNotFoundError). It might not be installed or in PATH.")
             return "Error: 'ls' command not found. Please ensure it is installed and in the system's PATH."
        except subprocess.TimeoutExpired:
             log.error(f"ls command timed out for path '{target_path}' after 15 seconds.")
             return f"Error: ls command timed out for path '{target_path}'."
        except Exception as e:
            log.exception(f"An unexpected error occurred while executing ls command for path '{target_path}': {e}")
            return f"An unexpected error occurred while executing ls: {str(e)}"

    # Assuming BaseTool provides a working get_function_declaration implementation