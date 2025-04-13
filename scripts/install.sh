#!/bin/bash
set -e

# Colors for output
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${BLUE}Installing Gemini Code...${RESET}"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed.${RESET}"
    echo "Please install pip and try again."
    exit 1
fi

# Install the package
echo -e "${YELLOW}Installing package from PyPI...${RESET}"
pip install --upgrade gemini-code

# Create shortcut in ~/.local/bin if it exists and is in PATH
if [[ -d "$HOME/.local/bin" && "$PATH" == *"$HOME/.local/bin"* ]]; then
    echo -e "${YELLOW}Creating 'gemini' shortcut...${RESET}"
    cat > "$HOME/.local/bin/gemini" << 'EOF'
#!/bin/bash
python -m gemini_cli "$@"
EOF
    chmod +x "$HOME/.local/bin/gemini"
fi

echo -e "${GREEN}Installation complete!${RESET}"
echo ""
echo -e "${BLUE}To get started:${RESET}"
echo -e "1. Set up your API key: ${YELLOW}gemini setup YOUR_GOOGLE_API_KEY${RESET}"
echo -e "2. Start using Gemini Code: ${YELLOW}gemini${RESET}"
echo ""
echo -e "For more information, visit: https://github.com/raizamartin/gemini-code"