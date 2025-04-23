# CLI Code Context Rules

This document explains how to customize the initial context used by the CLI Code assistant, using the `.rules` directory feature.

## Overview

The CLI Code assistant now supports a hierarchical approach to initializing context for each conversation:

1. First, it looks for a `.rules` directory and reads all Markdown (`.md`) files within it
2. If no `.rules` directory exists or it contains no Markdown files, it falls back to reading the project's `README.md` file
3. If neither of these sources is available, it uses the traditional `ls` output as a fallback

This feature allows you to provide more meaningful context to the assistant than just a directory listing, helping it understand your project's structure, guidelines, and conventions from the start.

## Using the `.rules` Directory

### Basic Setup

1. Create a `.rules` directory in your project's root:
   ```bash
   mkdir .rules
   ```

2. Create one or more Markdown (`.md`) files within this directory:
   ```bash
   touch .rules/context.md
   touch .rules/tools.md
   touch .rules/style.md
   ```

3. Each file should contain information that helps the assistant understand different aspects of your project.

### Recommended Structure

While you can organize your `.rules` files however you like, here's a recommended structure:

- **`context.md`**: General project overview, purpose, and key concepts
- **`structure.md`**: Project structure, important directories, and file organization
- **`style.md`**: Coding style, conventions, and best practices for the project
- **`tools.md`**: Development tools, build processes, and testing approaches
- **`patterns.md`**: Design patterns and architecture principles used in the project

### Example Content

Here's an example of what your `.rules/context.md` file might contain:

```markdown
# Project Context

This is a Python web application using Flask that provides an API for managing todo items.

## Key Features

- RESTful API for todos (CRUD operations)
- Authentication using JWT tokens
- PostgreSQL database backend
- Swagger documentation

## Important Files

- `app.py`: Main application entry point
- `models/`: Database models
- `routes/`: API endpoints
- `services/`: Business logic
- `tests/`: Test suite
```

## Benefits

Using the `.rules` directory provides several advantages:

1. **Better Understanding**: The assistant has immediate knowledge of your project's structure and conventions.
2. **Consistent Responses**: The assistant can provide more consistent responses aligned with your project guidelines.
3. **Reduced Repetition**: You don't need to repeatedly explain the same project details.
4. **Selective Context**: You can focus the context on what matters, rather than including all files in a directory listing.

## Best Practices

1. **Keep Files Focused**: Each file in the `.rules` directory should have a clear purpose.
2. **Be Concise**: While detailed information is helpful, avoid extremely long files as they consume context space.
3. **Update Regularly**: Keep your `.rules` files updated as your project evolves.
4. **Prioritize Important Information**: Put the most critical information in the beginning of each file.
5. **Use Markdown Features**: Utilize headings, lists, and code blocks to organize information.

## Visualization

The CLI Code assistant will inform you which context source it's using when you start a session:

```
Initializing provider gemini with model models/gemini-2.5-pro-exp-03-25...
Gemini model initialized successfully.
Context will be initialized from 3 .rules/*.md files.
```

This helps you confirm that your desired context is being used for the conversation.
