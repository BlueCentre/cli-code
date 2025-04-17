# Installing and Using CLI-Code

This guide explains how to install, configure, and use CLI-Code.

## Installation Options

### Option 1: Install from PyPI (Recommended)

```bash
uv pip install cli-code-agent
```

### Option 2: Install from Source

```bash
git clone https://github.com/BlueCentre/cli-code.git
cd cli-code
# Create a virtual environment (optional but recommended)
# uv venv
# source .venv/bin/activate

# Install in editable mode
uv pip install -e .
# For development including test dependencies, use:
# uv pip install -e '.[dev]'
```

## Setting Up Provider Credentials

Before using CLI-Code, you need to set up credentials for your chosen provider:

### For Gemini:

```bash
cli-code setup --provider=gemini YOUR_GOOGLE_API_KEY
```

To get a Google API key for Gemini:
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key or use an existing one
3. Copy the key and use it in the setup command above

### For Ollama:

```bash
cli-code setup --provider=ollama http://localhost:11434/v1
```

Make sure your Ollama server is running and accessible at the specified URL.

## Using CLI-Code

### Starting a Session

```bash
# Start with default provider and model
cli-code

# Start with a specific provider
cli-code --provider=ollama

# Start with a specific provider and model
cli-code --provider=gemini --model models/gemini-2.5-pro-exp-03-25
cli-code --provider=ollama --model llama3
```

### Setting Default Preferences

```bash
# Set default provider
cli-code set-default-provider gemini

# Set default model for a provider
cli-code set-default-model --provider=gemini models/gemini-2.5-pro-exp-03-25
cli-code set-default-model --provider=ollama llama3
```

### Listing Available Models

```bash
# List models for Gemini
cli-code list-models --provider=gemini

# List models for Ollama
cli-code list-models --provider=ollama
```

### Available Commands

During an interactive session, you can use these commands:

- `/help` - Display help information
- `/exit` - Exit the chat session

## Configuration Options

### Config File

Configuration is stored in `~/.config/cli-code/config.yaml` with this structure:

```yaml
google_api_key: YOUR_GEMINI_API_KEY
ollama_api_url: http://localhost:11434/v1

default_provider: gemini
default_model: models/gemini-2.5-pro-exp-03-25
ollama_default_model: llama3

settings:
  max_tokens: 1000000
  temperature: 0.5
  token_warning_threshold: 800000
  auto_compact_threshold: 950000
```

### Environment Variables

You can also use environment variables or a `.env` file:

```
CLI_CODE_GOOGLE_API_KEY=your_google_api_key_here
CLI_CODE_OLLAMA_API_URL=http://localhost:11434/v1
CLI_CODE_DEFAULT_PROVIDER=ollama
CLI_CODE_OLLAMA_DEFAULT_MODEL=llama3.2:latest
```

## Troubleshooting

If you encounter issues:

1. Verify your API credentials are correct: `cat ~/.config/cli-code/config.yaml`
2. Ensure you have a working internet connection
3. Check that you have Python 3.9+ installed: `python --version`
4. For Ollama, ensure the Ollama server is running: `curl http://localhost:11434/v1/models`

For more help, visit: https://github.com/BlueCentre/cli-code
