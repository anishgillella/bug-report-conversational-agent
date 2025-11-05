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
    solved: bool = Field(..., description="Whether bug is solved")


class ConversationOutput(BaseModel):
    """Final structured output from conversation."""
    success: bool = Field(..., description="Whether conversation gathered all required info")
    report: Optional[BugReport] = Field(None, description="Bug report if successful")


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
        self.solved: Optional[bool] = None
        
        # Conversation trace for logging
        self.trace: List[Dict[str, Any]] = []
    
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

Guidelines:
- Be concise and professional
- Ask one question at a time
- When a developer mentions a bug or work, extract the key information
- Do NOT try to help solve the bug - you're just gathering reports
- Be skeptical of vague responses and ask for clarification
- Confirm the final status (solved or not) before concluding
- When confirming a partial name match, show the suggestion clearly and wait for explicit confirmation
- When asking for bug selection, be clear and explicit - don't assume they'll pick one automatically

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
            
            # Check if this is the last turn before limit
            if self.turn_count >= self.max_turns:
                # Extract any final information from this last message
                self._extract_information()
                
                # Give a graceful exit message
                exit_msg = f"Thank you for your input! We've reached the conversation limit of {self.max_turns} turns. I'm now preparing your final report based on our discussion."
                print(f"\nBot: {exit_msg}\n")
                self.messages.append({"role": "assistant", "content": exit_msg})
                self.trace.append({"type": "message", "role": "assistant", "content": exit_msg})
                break
            
            # Get bot response
            bot_response = self.get_bot_response()
            print(f"\nBot: {bot_response}\n")
            self.trace.append({"type": "message", "role": "assistant", "content": bot_response})
            
            # Check for conversation completion signals
            self._extract_information()
            
            if self._is_conversation_complete():
                # Give final summary before ending
                farewell_msg = "Thank you for your detailed update! I've successfully gathered all the necessary information. Your report is now ready to be submitted to the bug tracking system."
                print(f"\nBot: {farewell_msg}\n")
                self.messages.append({"role": "assistant", "content": farewell_msg})
                self.trace.append({"type": "message", "role": "assistant", "content": farewell_msg})
                break
    
    def _extract_information(self):
        """
        Extract key information from recent messages.
        Looks for patterns like bug numbers, developer names, and status indicators.
        """
        # Simple pattern-based extraction for now
        if not self.messages:
            return
        
        # Check recent user messages for developer name
        if not self.developer_name:
            for msg in reversed(self.messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "").lower()
                    # Check against known developer names
                    for dev in self.data_manager.developers:
                        if dev["name"].lower() in content:
                            self.developer_name = dev["name"]
                            self.developer_id = dev["developer_id"]
                            break
                    if self.developer_name:
                        break
        
        # Check for bug selection patterns (Bug #X or just number)
        if not self.selected_bug_id:
            # First try to find "Bug #X" pattern
            conv_text = " ".join([msg.get("content", "") for msg in self.messages])
            bug_pattern = r"[Bb]ug\s*#?(\d+)"
            matches = re.findall(bug_pattern, conv_text)
            
            if matches:
                # Get the most recent bug number
                self.selected_bug_id = int(matches[-1])
            else:
                # If no "Bug" mention, look for standalone numbers in recent user messages
                for msg in reversed(self.messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "").strip()
                        # Check if message is just a number (bug ID)
                        if content.isdigit():
                            potential_bug_id = int(content)
                            # Validate it's a real bug ID
                            if self.data_manager.get_bug_by_id(potential_bug_id):
                                self.selected_bug_id = potential_bug_id
                                break
        
        # Check for solved status patterns
        if self.solved is None:
            # Look at ALL user messages to find solved status
            recent_user_messages = []
            for msg in reversed(self.messages):
                if msg.get("role") == "user":
                    recent_user_messages.append(msg.get("content", "").lower())
                    if len(recent_user_messages) >= 3:  # Get last 3 user messages
                        break
            
            recent_text = " ".join(recent_user_messages)
            
            # First check for explicit "not solved" phrases
            not_solved_indicators = ["not solved", "still working", "more work", "not done", "not finished", "still in progress"]
            for indicator in not_solved_indicators:
                if indicator in recent_text:
                    self.solved = False
                    return
            
            # Then check for solved indicators
            solved_indicators = ["solved", "fixed", "resolved", "done", "works", "working", "success"]
            for indicator in solved_indicators:
                if indicator in recent_text:
                    self.solved = True
                    return
            
            # If user just says "Yes" in context of being asked if solved
            if recent_text.strip() in ["yes", "yep", "yeah", "yup", "correct", "right"]:
                # This is likely confirming a previous status question
                if "not solved" not in recent_text and "still" not in recent_text:
                    self.solved = True
        
        # Collect progress notes from user messages about work done
        if not self.progress_note:
            for msg in reversed(self.messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    # Look for substantive content that's not just answers
                    if len(content) > 20 and not content.lower().startswith("i'm"):
                        self.progress_note = content
                        break
    
    def _is_conversation_complete(self) -> bool:
        """
        Check if we've gathered enough information to complete the conversation.
        Returns True when all required fields are collected and user indicates they're done.
        Uses LLM reasoning to detect natural conversation endings.
        """
        # Check if we have all required information
        has_all_fields = (
            self.developer_id is not None and
            self.selected_bug_id is not None and
            self.progress_note is not None and
            self.solved is not None
        )
        
        # Also check if we've had enough turns to gather info
        # (at least 4 turns: name, bug selection, work description, solved status)
        has_enough_turns = self.turn_count >= 4
        
        if not (has_all_fields and has_enough_turns):
            return False
        
        # Use LLM to reason about whether conversation should end
        # Get recent conversation context
        recent_messages = self.messages[-6:] if len(self.messages) > 6 else self.messages
        
        completion_prompt = """Based on this conversation, should we end the conversation and submit the bug report?
        
Consider: Does the user seem to be done reporting? Have they given all necessary information about their bug work?

Answer with only 'YES' or 'NO'."""
        
        # Create a temporary message list for the completion check
        temp_messages = recent_messages + [{"role": "user", "content": completion_prompt}]
        
        try:
            messages_for_llm = [
                {"role": "system", "content": "You are a brief analyzer. Answer only YES or NO."}
            ] + temp_messages
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_for_llm,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer
        except Exception as e:
            # If LLM call fails, fall back to simple heuristics
            # Just end if we have all fields and enough turns
            return True
    
    def get_structured_output(self) -> ConversationOutput:
        """
        Generate the final structured output by parsing the conversation.
        Uses Pydantic models for proper validation and structure.
        
        Returns a ConversationOutput object with validation.
        """
        # Extract any remaining information
        self._extract_information()
        
        # Determine if we have all required fields
        success = (
            self.developer_id is not None and
            self.selected_bug_id is not None and
            self.progress_note is not None and
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
