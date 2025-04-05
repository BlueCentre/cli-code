# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2]

- Significantly improved test coverage across multiple modules:
  - `config.py`: Increased to 89% coverage (from 73%)
  - `main.py`: Increased to 67% coverage (from 46%)
  - Overall test improvements bringing project coverage to over 75%
- Added comprehensive test files:
  - `test_config_comprehensive.py`: Tests for edge cases and advanced functionality
  - `test_main_comprehensive.py`: Tests for CLI commands and interactive features

## [0.2.1]

- Added Ollama provider support (see `ollama.py` model implementation)
- Added support for multiple model providers with configurable defaults
- Enhanced configuration system with hierarchical settings
- Added new CLI commands:
  - `set-default-provider` to select default provider
  - `set-default-model` to set default model for a provider
- Updated help text and documentation
- Configuration now supports environment variables via .env file
- Added test coverage improvements

## [0.1.21] 

- Implemented native Gemini function calling for much more reliable tool usage
- Rewritten the tool execution system to use Gemini's built-in function calling capability
- Enhanced the edit tool to better handle file creation and content updating
- Updated system prompt to encourage function calls instead of text-based tool usage
- Fixed issues with Gemini not actively creating or modifying files
- Simplified the BaseTool interface to support both legacy and function call modes

## [0.1.20]

- Fixed error with Flask version check in example code
- Improved error handling in system prompt example code

## [0.1.19]

- Improved system prompt to encourage more active tool usage
- Added thinking/planning phase to help Gemini reason about solutions
- Enhanced response format to prioritize creating and modifying files over printing code
- Filtered out thinking stages from final output to keep responses clean
- Made Gemini more proactive as a coding partner, not just an advisor

## [0.1.18]

- Updated default model to Gemini 2.5 Pro Experimental (models/gemini-2.5-pro-exp-03-25)
- Updated system prompts to reference Gemini 2.5 Pro
- Improved model usage and documentation

## [0.1.17]

- Added `list-models` command to show all available Gemini models
- Improved error handling for models that don't exist or require permission
- Added model initialization test to verify model availability
- Updated help documentation with new commands

## [0.1.16]

- Fixed file creation issues: The CLI now properly handles creating files with content
- Enhanced tool pattern matching: Added support for more formats that Gemini might use
- Improved edit tool handling: Better handling of missing arguments when creating files
- Added special case for natural language edit commands (e.g., "edit filename with content: ...")

## [0.1.15]

- Fixed tool execution issues: The CLI now properly processes tool calls and executes Bash commands correctly
- Fixed argument parsing for Bash tool: Commands are now passed as a single argument to avoid parsing issues
- Improved error handling in tools: Better handling of failures and timeouts
- Updated model name throughout the codebase to use `gemini-1.5-pro` consistently 