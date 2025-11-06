"""
Pydantic models for bug reporting chatbot.
Defines all data structures and validation schemas.
"""
from typing import List
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
    reports: List[BugReport] = Field(default_factory=list, description="List of bug reports from conversation")

