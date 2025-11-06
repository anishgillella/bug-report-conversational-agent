#!/bin/bash
# Bug Reporting Chatbot - Startup Script
# This script ensures all dependencies are loaded and runs the chatbot

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "✗ Error: .env file not found"
    echo "Please create .env with OPENROUTER_API_KEY and OPENROUTER_MODEL"
    exit 1
fi

# Load environment variables from .env
export $(cat .env | grep -v '^#' | xargs)

# Verify required variables are set
if [ -z "$OPENROUTER_API_KEY" ] || [ -z "$OPENROUTER_MODEL" ]; then
    echo "✗ Error: OPENROUTER_API_KEY or OPENROUTER_MODEL not set in .env"
    exit 1
fi

# Run the chatbot
python main.py

