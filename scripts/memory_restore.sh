#!/bin/bash
# memory_restore.sh - Restore assistant memory from backup
# Usage: ./scripts/memory_restore.sh

BACKUP_FILE="scripts/memory_backup.json"

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

chmod +x "$0" 