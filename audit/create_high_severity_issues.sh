#!/bin/bash

# Find all high-severity core issues
FILES=$(grep -l "High" issues/core/*.md)

for file in $FILES; do
  # Extract title from the first line
  TITLE=$(head -n 1 "$file" | sed 's/^# //')
  
  # Extract body from the rest of the file
  BODY=$(tail -n +3 "$file")
  
  # Create the issue using GitHub CLI
  # Adding labels: bug, audit, high-severity
  echo "Creating issue for: $TITLE"
  gh issue create --title "$TITLE" --body "$BODY" --label "code-audit,codex"
done
