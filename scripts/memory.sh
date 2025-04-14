#!/bin/bash
# memory.sh - Backup and restore assistant memory
# Usage: 
#   ./scripts/memory.sh backup - Creates a backup of assistant memory
#   ./scripts/memory.sh restore - Provides instructions to restore memory

BACKUP_FILE="scripts/memory_backup.json"

function backup_memory() {
  echo "To create a memory backup:"
  echo "1. Ask the assistant: 'Please create a memory backup'"
  echo "2. The assistant will use mcp_memory_read_graph tool to read current memory"
  echo "3. The assistant will update $BACKUP_FILE with current memory contents"
  echo
  echo "You can also ask the assistant to add specific information to memory before backing up."
}

function restore_memory() {
  if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
  fi
  
  echo "This script will help you restore memory in a new assistant session."
  echo "Instructions:"
  echo "1. When starting a new chat with the assistant, paste the following instructions:"
  echo
  echo "------- COPY BELOW THIS LINE -------"
  echo "Please restore my memory backup from the scripts/memory_backup.json file"
  echo "Steps:"
  echo "1. Read the file content with read_file tool"
  echo "2. Parse the JSON content" 
  echo "3. Create entities and relations from the backup using memory tools"
  echo "4. Confirm when memory has been restored"
  echo "------- COPY ABOVE THIS LINE -------"
  echo
  echo "The assistant will then be able to restore its memory from the backup file."
}

case "$1" in
  backup)
    backup_memory
    ;;
  restore)
    restore_memory
    ;;
  *)
    echo "Usage: $0 {backup|restore}"
    echo "  backup  - Provides instructions for backing up assistant memory"
    echo "  restore - Provides instructions for restoring assistant memory"
    exit 1
    ;;
esac 