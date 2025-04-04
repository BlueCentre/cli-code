"""
Tool for displaying directory structure using the 'tree' command.
"""

import logging
import os
import subprocess
from pathlib import Path

from .base import BaseTool

log = logging.getLogger(__name__)

DEFAULT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 10


class TreeTool(BaseTool):
    name: str = "tree"
    description: str = f"""Displays the directory structure as a tree. Shows directories and files.
        Use this to understand the hierarchy and layout of the current working directory or a subdirectory.
        Defaults to a depth of {DEFAULT_TREE_DEPTH}. Use the 'depth' argument to specify a different level.
        Optionally specify a 'path' to view a subdirectory instead of the current directory."""
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

    def execute(self, path: str | None = None, depth: int | str | None = None) -> str:
        """Executes the tree command."""

        if depth is None:
            depth_limit = DEFAULT_TREE_DEPTH
        else:
            # Convert depth to int if it's a string
            if isinstance(depth, str):
                try:
                    depth = int(depth)
                except ValueError:
                    log.warning(f"Invalid depth value '{depth}', using default {DEFAULT_TREE_DEPTH}")
                    depth = DEFAULT_TREE_DEPTH
                    
            # Clamp depth to be within reasonable limits
            depth_limit = max(1, min(depth, MAX_TREE_DEPTH))

        # Fix command construction to use proper list format
        command = ["tree", "-L", str(depth_limit)]

        # Add path if specified
        target_path = "."  # Default to current directory
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
                check=False,  # Don't raise exception on non-zero exit code
                timeout=15,  # Add a timeout
            )

            if process.returncode == 0:
                log.info(f"Tree command successful for path '{target_path}' with depth {depth_limit}.")
                # Limit output size? Tree can be huge.
                output = process.stdout.strip()
                if len(output.splitlines()) > 200:  # Limit lines as a proxy for size
                    log.warning(f"Tree output for '{target_path}' exceeded 200 lines. Truncating.")
                    output = "\n".join(output.splitlines()[:200]) + "\n... (output truncated)"
                return output
            elif process.returncode == 127 or "command not found" in process.stderr.lower():
                log.info("'tree' command not found. Falling back to Python-based implementation.")
                return self._fallback_tree_implementation(target_path, depth_limit)
            else:
                log.error(
                    f"Tree command failed with return code {process.returncode}. Path: '{target_path}', Depth: {depth_limit}. Stderr: {process.stderr.strip()}"
                )
                error_detail = process.stderr.strip() if process.stderr else "(No stderr)"
                log.info("Falling back to Python-based tree implementation.")
                return self._fallback_tree_implementation(target_path, depth_limit)

        except FileNotFoundError:
            log.error("'tree' command not found (FileNotFoundError). Falling back to Python implementation.")
            return self._fallback_tree_implementation(target_path, depth_limit)
        except subprocess.TimeoutExpired:
            log.error(f"Tree command timed out for path '{target_path}' after 15 seconds.")
            return (
                f"Error: Tree command timed out for path '{target_path}'. The directory might be too large or complex."
            )
        except Exception as e:
            log.exception(f"An unexpected error occurred while executing tree command for path '{target_path}': {e}")
            log.info("Attempting fallback to Python-based tree implementation.")
            try:
                return self._fallback_tree_implementation(target_path, depth_limit)
            except Exception as fallback_error:
                log.exception(f"Fallback tree implementation also failed: {fallback_error}")
                return f"An unexpected error occurred while displaying directory structure: {str(e)}"
                
    def _fallback_tree_implementation(self, path: str = ".", max_depth: int = DEFAULT_TREE_DEPTH) -> str:
        """
        A simple Python implementation of the tree command as a fallback 
        when the system tree command is not available.
        """
        log.info(f"Using Python-based tree fallback for path '{path}' with depth {max_depth}")
        
        try:
            result = []
            start_path = Path(path).resolve()
            
            if not start_path.exists():
                return f"Error: Path '{path}' does not exist."
            
            if not start_path.is_dir():
                return f"Error: Path '{path}' is not a directory."
                
            # Add the root directory to the output
            result.append(f".")
            
            # Walk the directory tree
            for root, dirs, files in os.walk(start_path):
                # Calculate current depth by counting path separators
                current_depth = len(Path(root).relative_to(start_path).parts)
                
                # Stop if we've reached max depth
                if current_depth >= max_depth:
                    dirs.clear()  # Don't go deeper
                    continue
                    
                # Sort directories and files for consistent output
                dirs.sort()
                files.sort()
                
                # Calculate prefix for current level
                indent = "│   " * current_depth
                
                # Add directories
                for i, dirname in enumerate(dirs):
                    is_last = (i == len(dirs) - 1 and not files)
                    prefix = "└── " if is_last else "├── "
                    result.append(f"{indent}{prefix}{dirname}/")
                
                # Add files
                for i, filename in enumerate(files):
                    is_last = (i == len(files) - 1)
                    prefix = "└── " if is_last else "├── "
                    result.append(f"{indent}{prefix}{filename}")
            
            # Limit output size
            if len(result) > 200:
                result = result[:200] + ["... (output truncated)"]
                
            return "\n".join(result)
            
        except Exception as e:
            log.exception(f"Error in fallback tree implementation: {e}")
            return f"Error generating directory tree: {str(e)}"
