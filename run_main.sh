#!/bin/bash

PYTHON_SCRIPT_DIR="../FedAVGBaseline"
PYTHON_SCRIPT_NAME="main.py"
OUTPUT_FILE="all_runs_output.txt"

for i in {1..10}
do
    echo "Run $i started..." >> "$OUTPUT_FILE"
    python "$PYTHON_SCRIPT_DIR/$PYTHON_SCRIPT_NAME" 2>&1 | tail -n 10 >> "$OUTPUT_FILE"
    echo "Run $i finished." >> "$PYTHON_SCRIPT_DIR/$OUTPUT_FILE"
done