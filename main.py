"""
Main entry point for the Bug Reporting Chatbot.
Handles interactive conversations with developers.
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from data_manager import DataManager
from llm_chat import BugReportingBot


def main():
    """Run the bug reporting chatbot."""
    # Load environment variables
    load_dotenv()
    
    # Initialize data manager
    try:
        data_manager = DataManager("data")
        print(f"✓ Loaded {len(data_manager.developers)} developers")
        print(f"✓ Loaded {len(data_manager.bugs)} bugs")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # Initialize bot
    try:
        bot = BugReportingBot(data_manager)
    except ValueError as e:
        print(f"✗ Error initializing bot: {e}")
        print("Make sure OPENROUTER_API_KEY and OPENROUTER_MODEL are set in .env")
        sys.exit(1)
    
    # Run interactive conversation
    bot.run_interactive()
    
    # Save conversation trace
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    trace_file = results_dir / f"trace_{len(list(results_dir.glob('trace_*.json'))) + 1}.json"
    with open(trace_file, "w") as f:
        json.dump(bot.trace, f, indent=2)
    
    print(f"\nConversation trace saved to: {trace_file}")
    
    # Save structured output attempt
    output = bot.get_structured_output()
    output_file = results_dir / f"output_{len(list(results_dir.glob('output_*.json'))) + 1}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
