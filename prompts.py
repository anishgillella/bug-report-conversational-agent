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

1. Ask for developer identification: "What is your developer ID or name?"
   - Accept any input: full name, partial name, ID number, or even misspellings
   - Be flexible and natural about matching
2. When user responds, use verify_developer tool with what they gave you
3. Based on the tool result, respond naturally:
   - If success: Confirm you found them and proceed
   - If similar match: Ask for confirmation ("Is this you?")
   - If multiple matches: Ask them to clarify which one
   - If no match: Show them valid names and ask again
4. Once developer is confirmed, IMMEDIATELY use get_bugs_for_developer to get their bugs
5. IMMEDIATELY show all bugs - do NOT wait for user input:
   Format each bug like:
   - **Bug ID:** X
   - **Description:** [description]
   - **Status:** [status]
   - **Solved:** Yes/No
   Then ask: "Which bug would you like to report on? (Please provide the Bug ID)"
6. Ask which bug to report on
7. Ask these questions ONE AT A TIME (wait for answer):
   - "What work have you done on this bug?" (accept any input)
   - "What is the current status? (Open, In Progress, Testing, Resolved, Closed)" 
     - If user gives multiple statuses or unclear answer, ask again: "Please choose ONE status"
   - VALIDATION: If status is "In Progress" and they say solved=Yes, ask for clarification:
     "You said it's 'In Progress' but also that it's solved. Did you mean 'Resolved'?"
   - "Is the bug now solved/working? (Yes/No)"
     - Only accept Yes or No, if unclear ask again
     - NOTE: If they say "Yes" but status is "In Progress", suggest "Resolved" status
8. Ask "Is there anything else that needs updating?"
9. If YES: Go back to step 6 for another bug
10. If NO: Say goodbye and end

IMPORTANT:
- Always show tool responses to user (name confirmations, suggestions, etc)
- Ask ONE question at a time
- Wait for the answer
- Never skip questions
- Never show summaries - just collect information
- Be natural but simple
- TRACK CONVERSATION LENGTH: After ~15 user responses, warn user that we're near the 20-turn limit
  Example: "Note: We're approaching the conversation limit. Please summarize any remaining updates."
- If conversation reaches 20 turns, wrap up gracefully and end

Just have a normal conversation. Don't worry about extracting or analyzing - the extraction happens later."""


class ExtractionPrompts:
    """SECOND LLM prompt - Extract structured data from conversation."""
    
    @staticmethod
    def get_final_analysis_prompt() -> str:
        """Second LLM prompt - Extract all bugs from conversation."""
        return """Analyze this conversation and extract ALL bug reports.

For EACH bug reported, extract EXACTLY:
- bug_id: number from conversation
- progress_note: user's work description - CHECK FOR SPELLING MISTAKES AND FIX THEM
  Examples: "conenction" → "connection", "wtih" → "with", "tqdm" → "tqdm"
  Fix common typos but keep technical terms as-is
- status: MUST be EXACTLY ONE of: Open, In Progress, Testing, Resolved, or Closed
  If user gave multiple statuses, pick the most recent/clear one
- solved: true if user said yes/solved/fixed, false if said no

LOGICAL VALIDATION RULES:
- If status is "Open", "In Progress", or "Testing" → solved MUST be false
- If status is "Resolved" or "Closed" → solved can be true or false
- If extracted data violates these rules, FIX IT (adjust status or solved accordingly)
  Example: If status="In Progress" and solved=true → change to solved=false
           OR change status to "Resolved" if the user clearly fixed it

SUCCESS DETERMINATION:
Analyze if the conversation was successful by checking:
- Did developer identify themselves? ✓
- Did they select a bug? ✓
- Did they provide progress notes? ✓
- Did they state bug status? ✓
- Set success=true if ALL above are true AND they provided meaningful information
- Set success=false if conversation was incomplete, vague, or developer not found

CONVERSATION:
{conversation_text}

Return ONLY a JSON object like this:
{{
  "success": true,
  "reports": [
    {{"bug_id": 1, "progress_note": "Fixed authentication", "status": "Resolved", "solved": true}},
    {{"bug_id": 5, "progress_note": "Added email queue", "status": "In Progress", "solved": false}}
  ]
}}

CRITICAL:
- FIX spelling mistakes in progress_note
- Use ONLY ONE status per bug (the clearest/most recent)
- ENFORCE LOGICAL CONSISTENCY between status and solved
- LLM determines success based on conversation quality, not just report count
- Extract from ACTUAL user responses only
- If a bug was discussed but not fully reported, skip it"""


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
