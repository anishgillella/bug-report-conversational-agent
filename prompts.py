"""
Simplified prompts for bug reporting chatbot.
FIRST LLM: Simple conversation - just ask questions in sequence.
SECOND LLM: Extract structured data from conversation.
"""
import json
from typing import List, Dict, Any


class ConversationPrompts:
    """SIMPLE prompts for first LLM - just ask questions in sequence."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """First LLM prompt - SIMPLE, mechanical conversation only."""
        return """You are a bug reporting assistant. Your ONLY job is to:

1. Ask for developer name
2. Use verify_developer tool to get their bugs
3. Show the bugs
4. Ask which bug to report on
5. Ask these questions ONE AT A TIME (wait for answer):
   - "What work have you done on this bug?"
   - "What is the current status? (Open, In Progress, Testing, Resolved, Closed)"
   - "Is the bug now solved/working? (Yes/No)"
6. Ask "Is there anything else that needs updating?"
7. If YES: Go back to step 4 for another bug
8. If NO: Say goodbye and end

IMPORTANT:
- Ask ONE question at a time
- Wait for the answer
- Never skip questions
- Never show summaries - just collect information
- Be natural but simple
- For tool calls: use verify_developer and get_bugs_for_developer

Just have a normal conversation. Don't worry about extracting or analyzing - the extraction happens later."""


class ExtractionPrompts:
    """SECOND LLM prompt - Extract structured data from conversation."""
    
    @staticmethod
    def get_final_analysis_prompt() -> str:
        """Second LLM prompt - Extract all bugs from conversation."""
        return """Analyze this conversation and extract ALL bug reports.

For EACH bug reported, extract EXACTLY:
- bug_id: number from conversation
- progress_note: exact words user said about work done
- status: what user said: Open, In Progress, Testing, Resolved, or Closed
- solved: true if user said yes/solved/fixed, false if said no

CONVERSATION:
{conversation_text}

Return ONLY a JSON array like this:
[
  {{"bug_id": 1, "progress_note": "Fixed authentication", "status": "Testing", "solved": true}},
  {{"bug_id": 5, "progress_note": "Added email queue", "status": "In Progress", "solved": false}}
]

Extract from ACTUAL user responses only. If a bug was discussed but not fully reported, skip it."""


class ToolDefinitions:
    """Tool definitions for LLM."""
    
    @staticmethod
    def get_tools() -> List[Dict[str, Any]]:
        """Define available tools for the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "verify_developer",
                    "description": "Verify that a developer exists in the system by name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The developer's name"
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bugs_for_developer",
                    "description": "Get all bugs assigned to a developer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "developer_id": {
                                "type": "integer",
                                "description": "The developer's ID"
                            }
                        },
                        "required": ["developer_id"]
                    }
                }
            }
        ]
