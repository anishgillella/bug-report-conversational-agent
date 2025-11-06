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
    
    # Get structured output
    output = bot.get_structured_output()
    
    # Update bug data for all completed reports
    for report in bot.completed_reports:
        update_success = data_manager.update_bug_progress(
            bug_id=report.bug_id,
            progress_note=report.progress_note,
            status=report.status,
            solved=report.solved
        )
        if update_success:
            print(f"✓ Updated Bug ID {report.bug_id} in database (Status: {report.status}, Solved: {report.solved})")
        else:
            print(f"✗ Failed to update Bug ID {report.bug_id}")
    
    # Display structured output to user
    print("\n" + "="*70)
    print("STRUCTURED OUTPUT - Ready for Bug Tracking System")
    print("="*70)
    print(json.dumps(output.model_dump(), indent=2))
    print("="*70 + "\n")
    
    # Save conversation trace
    results_dir = Path("results")
    traces_dir = results_dir / "traces"
    outputs_dir = results_dir / "outputs"
    traces_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    trace_file = traces_dir / f"trace_{len(list(traces_dir.glob('trace_*.json'))) + 1}.json"
    with open(trace_file, "w") as f:
        json.dump(bot.trace, f, indent=2)
    
    print(f"✓ Conversation trace saved to: {trace_file}")
    
    # Save structured output
    output_file = outputs_dir / f"output_{len(list(outputs_dir.glob('output_*.json'))) + 1}.json"
    with open(output_file, "w") as f:
        json.dump(output.model_dump(), f, indent=2)
    
    print(f"✓ Structured output saved to: {output_file}")


if __name__ == "__main__":
    main()
