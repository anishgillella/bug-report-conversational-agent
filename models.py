"""
Pydantic models for bug reporting chatbot.
Defines all data structures and validation schemas.
"""
from typing import Optional
from pydantic import BaseModel, Field


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


class ExtractedBugInfo(BaseModel):
    """Extracted bug information from conversation."""
    progress_note: Optional[str] = Field(None, description="Work done by user")
    status: Optional[str] = Field(None, description="Current bug status")
    solved: Optional[bool] = Field(None, description="Whether bug is solved")


class ConversationEndSignal(BaseModel):
    """Signal to determine if conversation should end."""
    should_end: bool = Field(..., description="Whether to end the conversation")
    reason: Optional[str] = Field(None, description="Why we should end or continue")

