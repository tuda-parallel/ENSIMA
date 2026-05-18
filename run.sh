#!/bin/bash

# Directory to watch
WATCH_DIR="./run"
DONE_DIR="$WATCH_DIR/done"

# Activate virtual environment
source ".venv/bin/activate"

# Create 'done' folder if it doesn't exist
mkdir -p "$DONE_DIR"

# Function to execute and move Python files
process_file() {
  local file="$1"
  echo "Executing $file..."
  python "$file"
  if [ $? -eq 0 ]; then
    mv "$file" "$DONE_DIR/"
    echo "Moved $file to $DONE_DIR"
  else
    echo "Error executing $file. Leaving it in place."
  fi
}

shopt -s nullglob
# 1) Process all Python files in the folder in order as expended:
# for file in "$WATCH_DIR"/*.py; do
#   process_file "$file"
# done

# 2) get the oldest file first
for file in $(ls -1tr "$WATCH_DIR"/*.py); do
  process_file "$file"
done

## 3) get newset file first
# for file in $(ls -t "$WATCH_DIR"/*.py); do
#   process_file "$file"
# done

shopt -u nullglob

echo "All Python scripts have been executed. Exiting."
