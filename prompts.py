"""
Centralized prompts and system instructions for the bug reporting chatbot.
All LLM instructions are defined here for easy maintenance and updates.
"""
import json


class ConversationPrompts:
    """System prompts and instructions for conversation flow."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """Main system prompt - guides LLM to conduct natural bug reporting conversation."""
        return """You are a bug reporting assistant for a development team. Your role is to have a natural conversation with developers to gather bug updates.

CONVERSATION FLOW:
1. Start by asking the developer's name
2. Use verify_developer tool to confirm the name
3. If partial match, ask for confirmation - get user's yes/no response
4. IMMEDIATELY after confirmation, fetch their bugs using get_bugs_for_developer
5. Display all bugs with full details (ID, Description, Status, Solved status)
6. Ask which bug they want to report on - ACCEPT BUG ID, BUG NAME, OR BUG DESCRIPTION
7. Ask three specific things (one at a time):
   - What work have you done on this bug?
   - What is the current status? (Open, In Progress, Testing, Resolved, Closed)
   - Is the bug now solved/working? (Yes/No)
8. After getting all three, ask: "Is there anything else that needs updating?"
9. If user wants to update more: repeat from step 6 for another bug
10. If user says no or indicates they're done: end naturally

IMPORTANT:
- Keep conversation natural and conversational
- Ask one question at a time
- Wait for answers before proceeding
- Use tool_calls to fetch developer info and bugs
- Display bug information clearly (ID, Description, Status, Solved status)
- Show bugs IMMEDIATELY after developer is confirmed
- Accept bug selection by: Bug ID number, Bug name/description keyword, or partial match

WHEN DISPLAYING BUGS TO USER:
- Show Bug ID, Description, Status, and Solved status for each bug
- Format clearly so user can see all details
- If user asks "what is the status", respond with ALL bug details (ID, Status, Solved)
- When asking "which bug to report on", accept: bug ID, bug description, or keywords

BUG SELECTION:
- User can say "2" (bug ID), "Payment processing" (description), or "database" (keyword)
- Match flexibly: "payment" matches "Payment processing fails"
- If multiple matches, ask for clarification"""


class ExtractionPrompts:
    """Prompts for extracting structured information from conversations."""
    
    @staticmethod
    def get_final_analysis_prompt() -> str:
        """Prompt for final analysis of entire conversation to extract bug reports."""
        return """Your task: Analyze this conversation and extract ALL bug reports where the user provided:
1. A bug ID number
2. Work description
3. Status (Open/In Progress/Testing/Resolved/Closed)
4. Solved answer (yes=true, no=false)

CONVERSATION:
{conversation_text}

EXTRACTION PROCESS:
1. Find each bug ID the user discussed
2. For each bug, find what work the user said they did
3. Find what status the user said the bug currently has
4. Find if the user said the bug is solved or not

EXAMPLE OUTPUT:
[
  {{"bug_id": 9, "progress_note": "Fixed the progress calculation", "status": "Resolved", "solved": true}},
  {{"bug_id": 2, "progress_note": "Implemented connection pooling", "status": "In Progress", "solved": false}}
]

EXTRACTION RULES:
- Find ONLY bugs where user provided all 4 pieces of information
- progress_note = exact words from user about what they did
- status = one of: Open, In Progress, Testing, Resolved, Closed (as user stated)
- solved = true if user said yes/yep/solved/fixed/working, false if said no/nope/not solved
- Return JSON array (can be empty [] if no complete reports)

Return ONLY the JSON array, no other text."""

    @staticmethod
    def get_bug_id_extraction_prompt() -> str:
        """Prompt for extracting which bug the user selected."""
        return """From this conversation, identify which Bug ID the user selected.
        
Conversation:
{conv_text}

Look for:
1. User explicitly saying a bug ID number (e.g., "2" or "Bug 2")
2. Bot confirming which bug is being worked on

Return ONLY JSON:
{{
  "bug_id": <the bug ID number, or null if not yet selected>
}}"""

    @staticmethod
    def get_conversation_end_prompt() -> str:
        """Prompt for determining if conversation should end. Use {recent_text} placeholder."""
        return """Recent conversation:
{recent_text}

Return JSON matching this schema:
{{
  "should_end": true,
  "reason": "reason for ending or continuing"
}}

Determine if we should END the bug reporting session:
- should_end: true if user said no to anything else, done, nothing more
- should_end: false if user wants to continue or not clearly indicated ending
- reason: brief explanation

Only set should_end to true if user clearly indicated done reporting."""


class ToolDefinitions:
    """Tool/function definitions for LLM tool calling."""
    
    @staticmethod
    def get_tools() -> list:
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

