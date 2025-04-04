# CLI Code Assistant - Architecture Analysis

This document outlines the architecture of the `cli-code` Python application, a command-line coding assistant powered initially by Google Gemini models, with planned support for other providers like Ollama.

> **Note on Diagrams**: All diagrams in this document follow [GitHub's Mermaid diagram specifications](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams) to ensure proper rendering on GitHub. Please refer to the GitHub documentation when making changes to any diagrams.

## 1. Overview

The application provides an interactive CLI experience where users can converse with a configured LLM (e.g., Gemini). The key feature is the model's ability to use local tools (file system operations, command execution, code analysis) via the provider's function calling mechanism (or equivalent) to fulfill user requests related to the codebase in the current working directory.

## 2. Core Components

The system is composed of several key Python modules and concepts:

```mermaid
graph TD
    CLI["CLI Frontend (main.py)"] --> Agent["Model Agent Abstraction (models/base.py)"];
    Agent --> GeminiAgent["Gemini Agent (models/gemini.py)"];
    Agent --> OllamaAgent["Ollama Agent (models/ollama.py)"];
    GeminiAgent --> GeminiAPI["Google Gemini API"];
    OllamaAgent --> OllamaAPI["Ollama OpenAI API"];
    Agent --> Tools["Tool Execution Layer (tools/)"];
    Tools --> FileSystem["Local System (Files/Shell)"];
    GeminiAPI --> GeminiAgent;
    OllamaAPI --> OllamaAgent;
```

*   **CLI (`src/cli_code/main.py`)**:
    *   **Responsibility**: User entry point, command parsing (`--provider`, `--model`), session initiation, basic output formatting. Instantiates the correct agent based on configuration/flags.
    *   **Technologies**: `click`, `rich`.
    *   **Interaction**: Takes user commands/input, initializes and invokes the appropriate `ModelAgent` instance.
*   **Configuration (`src/cli_code/config.py`)**:
    *   **Responsibility**: Loading and saving configuration data (API keys/URLs, default provider, default models).
    *   **Interaction**: Provides configuration values to the `CLI` and `ModelAgent` instances.
*   **Model Agent Abstraction (`src/cli_code/models/base.py`)** (Planned):
    *   **Responsibility**: Defines the common interface (`generate`, `list_models`) for all model provider implementations.
    *   **Interaction**: Serves as the base class for specific provider agents.
*   **Gemini Agent (`src/cli_code/models/gemini.py`)**:
    *   **Responsibility**: Implements the `AbstractModelAgent` interface for Google Gemini. Manages interaction with the Gemini API, orchestrates the agentic loop (prompting, function calling, tool execution), maintains conversation history, handles errors, and manages context. Implements the "Human-in-the-Loop" confirmation.
    *   **Technologies**: `google-generativeai`, `questionary`.
    *   **Interaction**: Communicates with the Gemini API, invokes tools via the `ToolRegistry`, receives tool results, formats final output for the `CLI`.
*   **Ollama Agent (`src/cli_code/models/ollama.py`)** (Planned):
    *   **Responsibility**: Implements the `AbstractModelAgent` interface for Ollama (via OpenAI compatible API). Handles API communication, tool schema translation, agentic loop, history, errors.
    *   **Technologies**: `openai` library, `questionary`.
    *   **Interaction**: Communicates with the Ollama API, invokes tools via the `ToolRegistry`, receives tool results, formats output.
*   **Tool Registry & Tools (`src/cli_code/tools/`)**:
    *   **Responsibility**: Defines the capabilities the agent can perform locally. Each tool provides its schema (adaptable for different providers). `__init__.py` acts as a registry.
    *   **Interaction**: Specific `ModelAgent` requests tool execution. Tools interact directly with the `Local System`.
*   **Utilities (`src/cli_code/utils.py`)**:
    *   **Responsibility**: Common helper functions.
    *   **Interaction**: Used by other components.
*   **External Services**:
    *   **Google Gemini API**: LLM service.
    *   **Ollama OpenAI API**: Alternative LLM service endpoint.
*   **Local System**:
    *   File System, Shell, Linters etc.

## 3. High-Level Interaction Flow (Agentic Loop)

The primary interaction follows an agentic loop within the active `ModelAgent.generate` method:

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI
    participant A as ActiveAgent
    participant P as ProviderAPI
    participant T as Tools

    U->>C: Input prompt (--provider=ollama)
    C->>A: Create OllamaAgent()
    C->>A: generate(prompt)
    A->>T: Get directory context
    T-->>A: Directory listing

    loop Agent Loop
        A->>P: generate_content() / completions.create()
        P-->>A: Response (Text or Function/Tool Call)

        alt is Function/Tool Call
            A->>A: Parse Call
            opt Requires Confirmation
                 A->>C: Request confirmation
                 C->>U: Ask user
                 U-->>C: Confirmation status
                 C-->>A: Status
                 alt User Rejects
                      A->>A: Handle rejection (inform API)
                      continue Loop
                 end
            end
            A->>T: Execute tool
            T-->>A: Tool result
            A->>A: Update history / Prepare tool response msg
        end
    end

    A->>C: Final Result
    C->>U: Display result
```

1.  **User Input**: User provides a prompt via `cli-code`, potentially with `--provider` and `--model` flags.
2.  **Agent Instantiation**: CLI determines the target provider (flag > config > default) and instantiates the corresponding `ModelAgent` (`GeminiModel` or `OllamaModel`).
3.  **Agent Invocation**: CLI calls the `Agent.generate` method.
4.  **Orientation**: Agent performs initial context gathering (e.g., `ls`).
5.  **LLM Call**: Agent sends the chat history and tool definitions (adapted for the provider's API) to the appropriate API (Gemini or Ollama).
6.  **LLM Response**: API responds with text or a request to call a tool/function.
7.  **Tool Execution (if requested)**:
    *   Agent parses the request.
    *   **Confirmation**: If needed, Agent uses `CLI` to ask the user.
    *   If confirmed/not needed, Agent retrieves the tool from `ToolRegistry`.
    *   Agent calls the tool's `execute` method.
    *   Tool interacts with the local system.
    *   Tool returns result to the Agent.
    *   Agent packages the result into the provider-specific format and adds it to history.
8.  **Loop Continuation**: Agent sends updated history back to the API (Step 5).
9.  **Task Completion**: Loop continues until the task is marked complete by the LLM (e.g., via text response or a dedicated signal like `task_complete`) or limits are reached.
10. **Final Output**: Agent returns the final response to the CLI.

## 4. C4 Model Diagrams

### Level 1: System Context

```mermaid
flowchart TD
    user["Developer\nUses the CLI to interact with their codebase via different LLM providers"]
    cli_app["CLI Code Assistant\nPython CLI application providing AI coding assistance with local tool usage, supporting multiple LLM backends"]
    gemini_api["Google Gemini API"]
    ollama_api["Ollama OpenAI API"]
    local_fs["Local File System"]
    local_shell["Local Shell"]
    local_tools["Local Dev Tools"]
    
    user --> |"Uses\n(CLI Commands/Prompts)"| cli_app
    cli_app --> |"Makes API calls\n(If provider=gemini)"| gemini_api
    cli_app --> |"Makes API calls\n(If provider=ollama)"| ollama_api
    cli_app --> |"Reads/Writes files/dirs"| local_fs
    cli_app --> |"Executes commands"| local_shell
    cli_app --> |"Invokes tools"| local_tools
    
    style user fill:#08427B,stroke:#052E56,color:#fff
    style cli_app fill:#1168BD,stroke:#0B4884,color:#fff
    style gemini_api fill:#999999,stroke:#6e6e6e,color:#fff
    style ollama_api fill:#999999,stroke:#6e6e6e,color:#fff
    style local_fs fill:#999999,stroke:#6e6e6e,color:#fff
    style local_shell fill:#999999,stroke:#6e6e6e,color:#fff
    style local_tools fill:#999999,stroke:#6e6e6e,color:#fff
```

### Level 2: Containers (Key Modules/Libraries)

```mermaid
flowchart TD
    user["Developer"]
    
    subgraph cli_system["CLI Code Assistant"]
        cli_main["CLI Frontend\nPython/Click/Rich\nHandles user commands, IO, selects and initializes agent"]
        agent_interface["Model Agent Interface\nPython/ABC\nDefines common agent behavior"]
        gemini_agent["Gemini Agent\nPython/google-generativeai\nImplements agent interface for Gemini"]
        ollama_agent["Ollama Agent\nPython/openai\nImplements agent interface for Ollama"]
        tools["Tool Execution Layer\nPython\nDefines and executes local actions"]
        config["Configuration\nYAML/File\nStores API keys/URLs, defaults"]
    end
    
    gemini_api["Google Gemini API"]
    ollama_api["Ollama OpenAI API"]
    local_fs["Local File System"]
    local_shell["Local Shell"]
    local_tools["Local Dev Tools"]
    
    user --> |"Uses"| cli_main
    
    cli_main --> |"Reads settings"| config
    cli_main --> |"Uses interface"| agent_interface
    cli_main -.-> |"Instantiates if provider=gemini"| gemini_agent
    cli_main -.-> |"Instantiates if provider=ollama"| ollama_agent
    
    gemini_agent --> |"Implements"| agent_interface
    ollama_agent --> |"Implements"| agent_interface
    
    gemini_agent --> |"Requests tool execution"| tools
    ollama_agent --> |"Requests tool execution"| tools
    gemini_agent --> |"Reads Gemini key"| config
    ollama_agent --> |"Reads Ollama URL"| config
    
    tools --> |"Returns results"| gemini_agent
    tools --> |"Returns results"| ollama_agent
    
    gemini_agent --> |"API Calls"| gemini_api
    ollama_agent --> |"API Calls"| ollama_api
    
    tools --> |"Accesses"| local_fs
    tools --> |"Accesses"| local_shell
    tools --> |"Accesses"| local_tools
    
    config --> |"Reads/Writes config file"| local_fs
    
    style user fill:#08427B,stroke:#052E56,color:#fff
    style cli_system fill:#444,stroke:#222,color:#fff
    style cli_main fill:#1168BD,stroke:#0B4884,color:#fff
    style agent_interface fill:#1168BD,stroke:#0B4884,color:#fff
    style gemini_agent fill:#1168BD,stroke:#0B4884,color:#fff
    style ollama_agent fill:#1168BD,stroke:#0B4884,color:#fff
    style tools fill:#1168BD,stroke:#0B4884,color:#fff
    style config fill:#1168BD,stroke:#0B4884,color:#fff
    style gemini_api fill:#999999,stroke:#6e6e6e,color:#fff
    style ollama_api fill:#999999,stroke:#6e6e6e,color:#fff
    style local_fs fill:#999999,stroke:#6e6e6e,color:#fff
    style local_shell fill:#999999,stroke:#6e6e6e,color:#fff
    style local_tools fill:#999999,stroke:#6e6e6e,color:#fff
```

## 5. Key Design Decisions & Patterns

*   **Provider Abstraction**: Using `AbstractModelAgent` to allow plugging in different LLM backends.
*   **Agentic Architecture**: The core logic resides in the specific `ModelAgent` implementations.
*   **Native Function/Tool Calling**: Leverages the respective provider's mechanism for tool use.
*   **Modular Tools**: Tools remain independent of the specific LLM provider.
*   **Explicit System Prompt**: Will likely need provider-specific system prompts tailored to their function calling nuances.
*   **Persistent History**: Maintained by the active agent instance.
*   **Human-in-the-Loop**: Confirmation logic remains in the agent, invoked before tool execution.
*   **Configuration Driven**: Provider and model selection controlled by config and CLI flags.
*   **Rich CLI**: `rich` and `questionary` enhance the user experience.
*   **Error Handling**: Needs to be robust within each agent implementation for provider-specific errors.

## 6. Potential Areas for Improvement

*   **Token-Based Context Management**: Implement for both providers.
*   **Sophisticated Planning**: Consider if needed beyond system prompts.
*   **Asynchronous Operations**: Evaluate for long-running tools.
*   **State Management**: Assess if needed for more complex multi-turn tasks.
*   **Testing**: Expand test suite (`test_dir`) to cover both providers, mock API interactions, and test the agent selection logic.
*   **Tool Schema Validation/Translation**: Ensure robust handling of schema differences between Gemini and OpenAI formats.
*   **Summarizer Tool Integration**: Clarify registration/usage.

## 7. Provider and Model Selection Logic

The CLI Code Assistant implements a sophisticated hierarchical resolution system for determining which provider and model to use. This system allows for flexibility in configuration while maintaining sensible defaults.

### Provider Selection

When determining which LLM provider to use, the application follows this precedence order (highest to lowest priority):

```mermaid
flowchart TD
    A[Start] --> B{CLI Flag?}
    B -->|Yes| C[Use provider from --provider flag]
    B -->|No| D{Environment Variable?}
    D -->|Yes| E[Use CLI_CODE_DEFAULT_PROVIDER]
    D -->|No| F{Config File Setting?}
    F -->|Yes| G[Use provider from config file]
    F -->|No| H[Use hardcoded default provider]
    
    C --> Z[Initialize selected provider]
    E --> Z
    G --> Z
    H --> Z
```

1. **Command-line flag**: If the user specifies `--provider=X`, that provider is used.
2. **Environment variable**: If `CLI_CODE_DEFAULT_PROVIDER` is set, that provider is used.
3. **Config file**: If a default provider is set in the configuration file, that provider is used.
4. **Hardcoded default**: If no other selection is found, the application falls back to the hardcoded default (currently "gemini").

### Model Selection

Once a provider is selected, the application determines which model to use for that provider using a similar precedence system:

```mermaid
flowchart TD
    A[Start] --> B{CLI Flag?}
    B -->|Yes| C[Use model from --model flag]
    B -->|No| D{Provider-specific Env Var?}
    D -->|Yes| E[Use provider-specific env var\ne.g., CLI_CODE_OLLAMA_DEFAULT_MODEL]
    D -->|No| F{Generic Model Env Var?}
    F -->|Yes| G[Use CLI_CODE_DEFAULT_MODEL]
    F -->|No| H{Config File Setting?}
    H -->|Yes| I[Use model from config file]
    H -->|No| J[Use provider's default model]
    
    C --> Z[Initialize provider with selected model]
    E --> Z
    G --> Z
    I --> Z
    J --> Z
```

1. **Command-line flag**: If the user specifies `--model=X`, that model is used.
2. **Provider-specific environment variable**: If a provider-specific environment variable is set (e.g., `CLI_CODE_OLLAMA_DEFAULT_MODEL`), that model is used.
3. **Generic model environment variable**: If `CLI_CODE_DEFAULT_MODEL` is set, that model is used.
4. **Config file**: If a default model is set in the configuration file for the selected provider, that model is used.
5. **Provider default**: If no other selection is found, the provider's default model is used (e.g., "gemini-2.5-pro-exp-03-25" for Gemini).

### Implementation

This logic is primarily implemented in `main.py` when processing CLI arguments and in `config.py` when loading configuration settings and environment variables. The environment variable loading system supports both direct environment variables and loading from a `.env` file, with direct environment variables taking precedence over `.env` file settings.

This hierarchical approach provides a balance between flexibility (users can easily override defaults) and convenience (sensible defaults mean minimal configuration required).

## 8. Context Management

The CLI Code Assistant implements a context management system to maintain coherent multi-turn conversations between the user and the LLM while operating within the constraints of each provider's context window limits.

### Current Implementation

#### Conversation History

Each Model Agent instance (e.g., GeminiModel, OllamaModel) maintains its own persistent conversation history throughout the session:

```mermaid
sequenceDiagram
    participant U as User
    participant A as Model Agent
    participant H as History
    participant LLM as LLM API
    
    U->>A: User prompt
    A->>H: Add user prompt
    A->>A: Add initial context (ls output)
    A->>H: Add context
    A->>LLM: Send history + tools
    LLM->>A: Response or tool call
    A->>H: Add model response
    
    alt is Tool Call
        A->>A: Execute tool
        A->>H: Add tool result
        A->>LLM: Send updated history
        LLM->>A: Next response
        A->>H: Add response
    end
    
    A->>U: Final response
```

The history structure includes:

1. **System Prompt**: Initially included to set the assistant's behavior.
2. **User Prompt**: Each user message enriched with contextual information (e.g., directory contents).
3. **Model Responses**: Text responses from the LLM.
4. **Tool Calls**: When the LLM decides to use a tool, the call is recorded.
5. **Tool Results**: Results of tool executions are added back to the history.

#### Initial Context Gathering

Before processing a user's request, the agent performs mandatory orientation:

1. By default, the agent executes an `ls` command to gather directory context.
2. This orientation helps the LLM understand the user's working environment.
3. The results are formatted and prepended to the user's actual prompt:

```
Current directory contents (from initial `ls`):
```
<directory listing>
```

User request: <actual user prompt>
```

#### Context Window Management

The current approach to handling context window constraints includes:

1. **History Management**: The `GeminiModel` class implements a `_manage_context_window` method that:
   - Keeps track of the number of turns in the conversation.
   - Limits history to `MAX_HISTORY_TURNS` (approximately 20 pairs of user/model interactions).
   - Preserves the initial system prompt and recent interactions.

2. **Provider-Specific Handling**: 
   - Each provider implementation (Gemini, Ollama) handles context structure according to the provider's API requirements.
   - For Gemini, the history is formatted as a list of message objects with "role" and "parts".
   - For Ollama (OpenAI-compatible), the history is formatted as a list of message objects with "role" and "content".

### Planned Enhancements

The following improvements to context management are planned:

#### Token-Aware Context Management

```mermaid
flowchart TD
    A[New Content] --> B[Token Counter]
    B --> C{Would exceed\ntoken limit?}
    C -->|Yes| D[Context Trimming]
    C -->|No| E[Add to Context]
    
    D --> F{Trimming strategy}
    F -->|Sliding Window| G[Remove oldest turns]
    F -->|Summarization| H[Summarize older turns]
    F -->|Hybrid| I[Summarize some, remove others]
    
    G --> E
    H --> E
    I --> E
```

1. **Token Counting**: Implement accurate token counting for both providers:
   - For Gemini, use the provider's token counting API.
   - For Ollama, implement or use an existing tokenizer.

2. **Dynamic Trimming**: Instead of a fixed number of turns:
   - Implement a sliding window based on token count rather than turn count.
   - Consider importance of turns (e.g., keep system prompt, recent interactions, and critical tool outputs).

3. **Content Compression**:
   - For lengthy tool outputs (e.g., `view` results of large files), implement summarization or truncation.
   - Consider indexing large file contents and only including relevant sections in the context.

4. **Context Summarization**:
   - For long-running sessions, summarize older turns rather than discarding them.
   - Potentially use the model itself to create summaries of previous interactions.

#### Context Persistence

1. **Session Recovery**:
   - Implement a mechanism to save and restore conversation context between sessions.
   - Allow users to continue previous conversations.

2. **Context Visualization**:
   - Provide commands to inspect the current context size.
   - Allow users to view and manage which parts of history are being retained.

This enhanced context management system will balance maintaining coherent multi-turn conversations with the token limitations of current LLM architectures, providing a more reliable and efficient user experience.

This analysis provides a comprehensive overview of the planned `cli-code` architecture. 