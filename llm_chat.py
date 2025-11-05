"""
LLM chat module for interacting with OpenRouter API.
Handles conversation state, tool calling, and structured output.
"""
import json
import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from data_manager import DataManager


# Pydantic models for structured output
class BugReport(BaseModel):
    """Structured bug report from conversation."""
    bug_id: int = Field(..., description="Unique bug identifier")
    progress_note: str = Field(..., description="Timestamped progress entry")
    status: str = Field(..., description="Bug status (Open, In Progress, Testing, Resolved, Closed)")
    solved: bool = Field(..., description="Whether bug is solved")


class ConversationOutput(BaseModel):
    """Final structured output from conversation."""
    success: bool = Field(..., description="Whether conversation gathered all required info")
    report: Optional[BugReport] = Field(None, description="Bug report if successful")


class ExtractedInfo(BaseModel):
    """Extracted information from conversation."""
    developer_id: Optional[int] = Field(None, description="Developer ID")
    bug_id: Optional[int] = Field(None, description="Bug ID")
    progress_note: Optional[str] = Field(None, description="Work performed on bug")
    status: Optional[str] = Field(None, description="Bug status (Open, In Progress, Testing, Resolved, Closed)")
    solved: Optional[bool] = Field(None, description="Whether bug is solved")


class BugReportingBot:
    """Conversational bot for bug reporting through OpenRouter LLM."""
    
    def __init__(self, data_manager: DataManager):
        """
        Initialize the bot with OpenRouter client and data manager.
        
        Args:
            data_manager: DataManager instance for accessing bug/developer data
        """
        self.data_manager = data_manager
        
        # Initialize OpenRouter client
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL")
        
        if not api_key or not model:
            raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL must be set in environment")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        
        # Conversation state
        self.messages: List[Dict[str, Any]] = []
        self.turn_count = 0
        self.max_turns = 20
        
        # Gathered information
        self.developer_id: Optional[int] = None
        self.developer_name: Optional[str] = None
        self.selected_bug_id: Optional[int] = None
        self.progress_note: Optional[str] = None
        self.status: Optional[str] = None  # Bug status: Open, In Progress, Testing, Resolved, Closed
        self.solved: Optional[bool] = None  # Whether bug is actually solved
        
        # Conversation trace for logging
        self.trace: List[Dict[str, Any]] = []
        
        # Track completed reports for multiple updates
        self.completed_reports: List[BugReport] = []
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.
        The prompt should NOT include the bug database.
        """
        return """You are a helpful bug reporting assistant for a software development team. Your role is to conduct brief conversations with developers to gather information about bug fixes and updates.

Your goal is to:
1. Identify the developer by their name (MUST verify using verify_developer tool first)
2. Ask them to SELECT ONE SPECIFIC BUG from their assigned bugs
3. Collect information about the work performed on that specific bug
4. Determine if that bug has been solved

IMPORTANT CONVERSATION SEQUENCE:
1. When you get a developer name, ALWAYS call verify_developer(name) FIRST
2. If verify_developer returns type="partial_match_needs_confirmation":
   - Show the tool's confirmation message to the user
   - WAIT for the user to confirm (yes/no)
   - Only proceed to get_bugs_for_developer if user confirms
   - If user denies, ask for the correct name
3. If verify_developer returns success=false:
   - Show the tool's error message with valid names
   - Ask user to provide the correct name
4. If verify_developer returns success=true:
   - Call get_bugs_for_developer to fetch their assigned bugs
   - Show the list of bugs with their IDs and descriptions
   - EXPLICITLY ask: "Which bug would you like to report on?" or "Which of these bugs would you like to provide an update for?"
   - WAIT for the user to select ONE bug by ID or description
   - CONFIRM which bug they selected
5. Only after bug selection is confirmed, collect work progress and status

BUG SELECTION IS CRITICAL:
   - Do NOT assume which bug they want to report on
   - Always ask explicitly for bug selection
   - Show bug IDs and short descriptions to make selection easy
   - Confirm their selection before proceeding
   - If they mention a bug that's not in their list, remind them of their assigned bugs

CRITICAL - INFORMATION TO COLLECT (ALL THREE REQUIRED):
You MUST collect exactly three pieces of information for each bug:
  1. PROGRESS NOTE: What work was done on the bug? (e.g., "Fixed the regex pattern", "Refactored database query")
     - Ask: "What work have you done on this bug?" or "Can you describe the work you performed?"
     - This is REQUIRED - do NOT skip it
  2. STATUS: The workflow state (Open, In Progress, Testing, Resolved, Closed)
     - Ask: "What is the current status of Bug #X?"
     - This is REQUIRED - do NOT skip it
  3. SOLVED: Whether the bug is functionally fixed (true/false)
     - Ask: "Is this bug now solved/working?" (expecting: Yes/No)
     - This is REQUIRED - do NOT skip it

These are THREE DIFFERENT things:
  * PROGRESS NOTE: Describes what work was done
  * STATUS: Describes workflow state (where it is in the pipeline)
  * SOLVED: Describes if it's functionally fixed (boolean)

SEQUENCE AFTER BUG SELECTION:
1. Ask for PROGRESS NOTE: "What work have you done on this bug?"
2. Ask for STATUS: "What is the current status?"
3. Ask for SOLVED: "Is this bug now solved/working?"

Guidelines:
- Be concise and professional
- Ask one question at a time
- ALWAYS ask for progress notes (work done) - this is NOT optional
- When a developer mentions a bug or work, extract the key information
- If the developer asks questions about a bug (e.g., "What was this bug about?"), answer their questions to help them provide context
- Do NOT try to help solve the bug - you're just gathering reports and answering clarification questions
- Be skeptical of vague responses and ask for clarification
- REQUIRE explicit confirmation for solved status - don't infer it from work descriptions
- When confirming a partial name match, show the suggestion clearly and wait for explicit confirmation
- When asking for bug selection, be clear and explicit - don't assume they'll pick one automatically
- Never skip asking for: progress notes, status, OR solved status

Keep responses natural and conversational."""
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """
        Define the tools available to the LLM.
        Returns OpenAI-compatible tool definitions.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "verify_developer",
                    "description": "Verify that a developer exists in the system and get their ID. Use this first to confirm the developer before getting their bugs.",
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
                    "description": "Fetch all bugs assigned to a developer. Use this after verifying the developer exists.",
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
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool and return the result as a string.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Arguments for the tool
            
        Returns:
            JSON string of tool result
        """
        if tool_name == "verify_developer":
            name = tool_input.get("name")
            dev = self.data_manager.find_developer_by_name(name)
            
            if dev and dev["name"].lower() == name.lower():
                # Exact full name match - verified immediately
                self.developer_id = dev["developer_id"]
                self.developer_name = dev["name"]
                return json.dumps({
                    "success": True,
                    "developer_id": dev["developer_id"],
                    "name": dev["name"],
                    "message": f"Developer {dev['name']} verified successfully",
                    "type": "exact_match"
                })
            elif dev:
                # Partial match found - ask for confirmation
                self.developer_id = dev["developer_id"]
                self.developer_name = dev["name"]
                return json.dumps({
                    "success": True,
                    "developer_id": dev["developer_id"],
                    "name": dev["name"],
                    "message": f"Partial match found. Please confirm: Did you mean {dev['name']}?",
                    "type": "partial_match_needs_confirmation",
                    "confirmation_required": True
                })
            else:
                # Check for partial matches (multiple possible matches)
                similar = self.data_manager.find_similar_developers(name)
                
                if similar:
                    # Found potential matches - ask for confirmation
                    similar_names = [d["name"] for d in similar]
                    return json.dumps({
                        "success": False,
                        "message": f"'{name}' is ambiguous",
                        "potential_matches": similar_names,
                        "suggestion": f"Did you mean one of these? {', '.join(similar_names)}"
                    })
                else:
                    # No matches at all - show all valid developers
                    valid_names = [d["name"] for d in self.data_manager.developers]
                    return json.dumps({
                        "success": False,
                        "message": f"Developer '{name}' not found in system",
                        "valid_developers": valid_names
                    })
        
        elif tool_name == "get_bugs_for_developer":
            dev_id = tool_input.get("developer_id")
            bugs = self.data_manager.get_bugs_for_developer(dev_id)
            
            # Format bugs for display (don't include all details to keep context small)
            formatted_bugs = [
                {
                    "bug_id": bug["bug_id"],
                    "description": bug["description"],
                    "status": bug["status"],
                    "solved": bug["solved"]
                }
                for bug in bugs
            ]
            
            return json.dumps(formatted_bugs)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    def add_user_message(self, text: str):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": text})
    
    def get_bot_response(self) -> str:
        """
        Get the bot's response using OpenRouter LLM.
        Handles tool calling and returns the final text response.
        """
        # Prepare messages with system prompt
        messages_with_system = [
            {"role": "system", "content": self._get_system_prompt()}
        ] + self.messages
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages_with_system,
            tools=self._get_tools(),
            max_tokens=500
        )
        
        # Process response
        choice = response.choices[0]
        
        # Handle tool calls if present
        while choice.message.tool_calls:
            # Add assistant's response with tool calls to messages
            assistant_msg = {"role": "assistant", "content": choice.message.content or ""}
            if hasattr(choice.message, 'tool_calls'):
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in choice.message.tool_calls
                ]
            self.messages.append(assistant_msg)
            
            # Handle each tool call
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                
                # Log tool call in trace
                self.trace.append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_input
                })
                
                # Execute tool
                tool_result = self._execute_tool(tool_name, tool_input)
                
                # Add tool result to messages - use correct format
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # Get another response after tool execution
            messages_with_system = [
                {"role": "system", "content": self._get_system_prompt()}
            ] + self.messages
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_with_system,
                tools=self._get_tools(),
                max_tokens=500
            )
            choice = response.choices[0]
        
        # Extract final text response
        bot_text = choice.message.content or ""
        
        # Add to messages for context
        if bot_text:
            self.messages.append({"role": "assistant", "content": bot_text})
        
        return bot_text
    
    def run_interactive(self):
        """Run an interactive conversation with the user."""
        print("\n" + "="*60)
        print("BUG REPORTING CHATBOT")
        print("="*60)
        print("Type 'quit' to exit conversation\n")
        
        # Initial bot greeting
        initial_msg = "Hi! Welcome to the bug reporting system. I'm here to help you report progress on bugs you've been working on. What's your name?"
        print(f"Bot: {initial_msg}\n")
        
        self.messages.append({"role": "assistant", "content": initial_msg})
        self.trace.append({"type": "message", "role": "assistant", "content": initial_msg})
        
        while self.turn_count < self.max_turns:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() == "quit":
                print("\nConversation ended by user.")
                break
            
            if not user_input:
                continue
            
            self.turn_count += 1
            
            # Add user message and log in trace
            self.add_user_message(user_input)
            self.trace.append({"type": "message", "role": "user", "content": user_input})
            
            # Check if we have a complete report (developer, bug, progress, status, AND explicit solved status)
            has_complete_report = (
                self.developer_id is not None and
                self.selected_bug_id is not None and
                self.progress_note is not None and
                self.status is not None and
                self.solved is not None
            )
            
            # If we have a complete report, save it for later
            if has_complete_report and self.selected_bug_id not in [r.bug_id for r in self.completed_reports]:
                # Create timestamped progress note
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                progress_note_with_timestamp = f"{timestamp} - {self.progress_note}"
                
                report = BugReport(
                    bug_id=self.selected_bug_id,
                    progress_note=progress_note_with_timestamp,
                    status=self.status,
                    solved=self.solved
                )
                self.completed_reports.append(report)
                self._reset_for_next_report()
                
                # The LLM will naturally ask if user has more bugs in its next response
                # No separate LLM call needed
            
            # Check if user wants to end conversation BEFORE generating bot response
            should_end = self._should_end_conversation()
            
            if should_end:
                # Generate confirmation summary of all detected information
                confirmation_msg = self._generate_confirmation_summary()
                print(f"\nBot: {confirmation_msg}\n")
                self.messages.append({"role": "assistant", "content": confirmation_msg})
                self.trace.append({"type": "message", "role": "assistant", "content": confirmation_msg})
                
                # Wait for user confirmation
                print("You: ", end="", flush=True)
                confirmation_input = input().strip()
                
                if not confirmation_input:
                    continue
                
                self.turn_count += 1
                self.add_user_message(confirmation_input)
                self.trace.append({"type": "message", "role": "user", "content": confirmation_input})
                
                # Check if user confirmed
                is_confirmed = self._check_confirmation(confirmation_input)
                
                if is_confirmed:
                    # User confirmed, ask if they want to update more bugs
                    more_bugs_msg = "Great! Would you like to update any other bugs, or are we all set?"
                    print(f"\nBot: {more_bugs_msg}\n")
                    self.messages.append({"role": "assistant", "content": more_bugs_msg})
                    self.trace.append({"type": "message", "role": "assistant", "content": more_bugs_msg})
                    
                    # Wait for user response
                    print("You: ", end="", flush=True)
                    more_input = input().strip()
                    
                    if not more_input:
                        continue
                    
                    self.turn_count += 1
                    self.add_user_message(more_input)
                    self.trace.append({"type": "message", "role": "user", "content": more_input})
                    
                    # Check if user wants to continue
                    wants_more = self._check_wants_more_bugs(more_input)
                    
                    if not wants_more:
                        # User is done, show final JSON and exit
                        farewell_msg = "Perfect! Your bug reports are now ready to be submitted to the bug tracking system."
                        print(f"\nBot: {farewell_msg}\n")
                        self.messages.append({"role": "assistant", "content": farewell_msg})
                        self.trace.append({"type": "message", "role": "assistant", "content": farewell_msg})
                        break
                    else:
                        # User wants to update more bugs, reset state and continue
                        # Reset developer_id to allow picking another developer's bug or same developer
                        self.selected_bug_id = None
                        self.progress_note = None
                        self.status = None
                        self.solved = None
                        
                        # Ask which bug they want to report on
                        continue_msg = f"Great! Let's update another bug. Which bug would you like to report on next?"
                        print(f"\nBot: {continue_msg}\n")
                        self.messages.append({"role": "assistant", "content": continue_msg})
                        self.trace.append({"type": "message", "role": "assistant", "content": continue_msg})
                        continue
                else:
                    # User wants to make changes, continue conversation to fix data
                    correction_msg = "No problem! What would you like to correct or add?"
                    print(f"\nBot: {correction_msg}\n")
                    self.messages.append({"role": "assistant", "content": correction_msg})
                    self.trace.append({"type": "message", "role": "assistant", "content": correction_msg})
                    
                    # Reset to allow re-collecting information
                    self.selected_bug_id = None
                    self.progress_note = None
                    self.status = None
                    self.solved = None
                    continue
            
            # Check if this is the last turn before limit
            if self.turn_count >= self.max_turns:
                # Give a graceful exit message
                exit_msg = f"Thank you for your input! We've reached the conversation limit of {self.max_turns} turns. I'm now preparing your final report based on our discussion."
                print(f"\nBot: {exit_msg}\n")
                self.messages.append({"role": "assistant", "content": exit_msg})
                self.trace.append({"type": "message", "role": "assistant", "content": exit_msg})
                break
            
            # Get bot response - let the LLM have the conversation naturally
            bot_response = self.get_bot_response()
            print(f"\nBot: {bot_response}\n")
            self.messages.append({"role": "assistant", "content": bot_response})
            self.trace.append({"type": "message", "role": "assistant", "content": bot_response})
            
            # Extract information from the conversation after getting bot response
            self._extract_info_from_response(bot_response)
    
    def _extract_information(self):
        """
        Extract information from the conversation using a single LLM call during get_bot_response.
        This method is now a no-op since extraction happens in the main LLM call.
        """
        pass
    
    def _extract_info_from_response(self, response_text: str):
        """
        Extract information from the last bot response using the LLM.
        This is called after getting a bot response to avoid extra LLM calls.
        """
        extraction_prompt = """Based on this conversation, extract the following information in JSON format:
{
  "developer_id": <number or null>,
  "bug_id": <number or null>,
  "progress_note": "<string describing ONLY the work done on the bug, NOT the status>",
  "status": "<string or null>",
  "solved": <true/false or null>
}

CRITICAL INSTRUCTIONS:
1. PROGRESS_NOTE: Extract ONLY what work the developer did or what they found
   - Examples: "Fixed memory leak", "Added caching", "Optimized queries", "Found root cause"
   - DO NOT include status words like "In Progress", "Resolved", "Open"
   - DO NOT include answers to "is it solved" (Yes/No answers belong in 'solved' field)

2. STATUS: Extract the workflow state (Open, In Progress, Testing, Resolved, Closed)
   - Find LATEST status mentioned

3. SOLVED: Extract whether bug functionally works (true/false only)
   - This is INDEPENDENT from status

Return ONLY valid JSON, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages + [{"role": "user", "content": extraction_prompt}],
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            data = json.loads(result_text)
            
            # Validate with Pydantic model
            extracted = ExtractedInfo(**data)
            
            # Update fields
            if extracted.developer_id and not self.developer_id:
                self.developer_id = extracted.developer_id
            
            if extracted.bug_id and not self.selected_bug_id:
                self.selected_bug_id = extracted.bug_id
            
            if extracted.progress_note and not self.progress_note:
                self.progress_note = extracted.progress_note
            
            if extracted.status:
                self.status = extracted.status
            
            if extracted.solved is not None:
                self.solved = extracted.solved
                
        except Exception as e:
            pass
    
    def _generate_confirmation_summary(self) -> str:
        """Generate a summary of all detected bug reports for user confirmation."""
        if not self.completed_reports:
            return "I don't have any bug reports to confirm yet. Could you provide details about the bugs you've been working on?"
        
        summary_lines = ["Let me confirm what I've detected:\n"]
        
        for i, report in enumerate(self.completed_reports, 1):
            summary_lines.append(f"**Bug #{i}:**")
            summary_lines.append(f"  - Bug ID: {report.bug_id}")
            summary_lines.append(f"  - Status: {report.status}")
            summary_lines.append(f"  - Solved: {report.solved}")
            summary_lines.append(f"  - Work Done: {report.progress_note.split(' - ', 1)[1] if ' - ' in report.progress_note else report.progress_note}")
            summary_lines.append("")
        
        summary_lines.append("Is this information correct? (Yes/No)")
        return "\n".join(summary_lines)
    
    def _check_confirmation(self, user_response: str) -> bool:
        """Check if user confirmed the summary."""
        response_lower = user_response.lower().strip()
        
        # Positive confirmations
        if any(word in response_lower for word in ["yes", "yep", "yeah", "correct", "that's right", "looks good", "perfect"]):
            return True
        
        # Negative responses
        if any(word in response_lower for word in ["no", "nope", "incorrect", "wrong", "not correct"]):
            return False
        
        # Default to no if unclear
        return False
    
    def _check_wants_more_bugs(self, user_response: str) -> bool:
        """Check if user wants to update more bugs."""
        response_lower = user_response.lower().strip()
        
        # Positive responses (wants more bugs)
        if any(word in response_lower for word in ["yes", "yep", "yeah", "another", "more", "update", "continue"]):
            return True
        
        # Negative responses (all set)
        if any(word in response_lower for word in ["no", "nope", "done", "that's it", "all set", "we're done", "that's all"]):
            return False
        
        # Default to no (assume they're done)
        return False
    
    def _reset_for_next_report(self) -> None:
        """Reset state after a successful report for the next bug report."""
        self.selected_bug_id = None
        self.progress_note = None
        self.status = None
        self.solved = None
    
    def _should_end_conversation(self) -> bool:
        """
        Check if the user has indicated they want to end the conversation.
        This is a separate check from the main conversation.
        """
        # Get the last user message
        last_user_msg = None
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").strip().lower()
                break
        
        # Don't end if no recent user message
        if not last_user_msg:
            return False
        
        # Only end if we have at least one completed report and user says they're done
        # Don't require all fields to be set - user might be done after first report
        if not self.completed_reports:
            return False
        
        # Simple direct check first - common end signals
        end_signals = ["no", "nope", "that's it", "that is it", "i'm done", "i am done", "no more", "nothing else", "nothing", "done"]
        if any(signal in last_user_msg for signal in end_signals):
            return True
        
        # If not a simple signal, use LLM for context-aware decision
        # Get recent conversation context for better understanding
        recent_msgs = []
        for msg in self.messages[-4:]:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")[:100]  # Truncate for clarity
            recent_msgs.append(f"{role}: {content}")
        
        context = "\n".join(recent_msgs)
        
        # Ask a separate LLM to determine if user wants to end
        end_check_prompt = f"""Recent conversation:
{context}

Does the user's most recent message indicate they want to END and submit the report?
Answer with ONLY 'YES' or 'NO'."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": end_check_prompt}],
                max_tokens=5
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer
        except Exception:
            return False
    
    def get_structured_output(self) -> ConversationOutput:
        """
        Generate the final structured output by parsing the conversation.
        Uses Pydantic models for proper validation and structure.
        
        Returns a ConversationOutput object with validation.
        """
        # If we have completed reports, return the first one
        if self.completed_reports:
            return ConversationOutput(
                success=True,
                report=self.completed_reports[0]  # Return first report
            )
        
        # Fallback: extract any remaining information
        self._extract_information()
        
        # Determine if we have all required fields
        success = (
            self.developer_id is not None and
            self.selected_bug_id is not None and
            self.progress_note is not None and
            self.status is not None and
            self.solved is not None
        )
        
        if success:
            # Verify the bug belongs to this developer
            bug = self.data_manager.get_bug_by_id(self.selected_bug_id)
            if bug and bug["assigned_dev"] == self.developer_id:
                # Create timestamped progress note
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                progress_note_with_timestamp = f"{timestamp} - {self.progress_note}"
                
                # Create validated Pydantic objects
                report = BugReport(
                    bug_id=self.selected_bug_id,
                    progress_note=progress_note_with_timestamp,
                    status=self.status,
                    solved=self.solved
                )
                
                return ConversationOutput(
                    success=True,
                    report=report
                )
            else:
                # Bug not assigned to this developer
                return ConversationOutput(
                    success=False,
                    report=None
                )
        else:
            # Not all required fields collected
            return ConversationOutput(
                success=False,
                report=None
            )
