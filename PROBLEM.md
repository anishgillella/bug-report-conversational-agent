# Quintess AI Take-Home Test - Bug resolution chatbot

## Overview
This test is designed to evaluate your skills in conversational AI and software engineering. You will be building a chatbot system for a bug tracking platform that gathers information from developers about bugs they've been working on.

## Problem Statement

### Your Task
Build a chatbot system that guides developers through a natural conversation flow to update bug progress, doing the following

- identify the developer and the bug on which they are reporting
- gather information about the work done on the bug and whether the bug is solved
- produce a structured output from the conversation to enter into the bug tracking system

Important: this is not a troubleshooting assistant, it should not try to help the developer solve the bug, only report on it.

### Requirements

#### Conversation Flow
The bot should interview the developer following these steps, sequentially

1. **Developer Identification**: Identify developers by their name, from the dataset.
2. **Bug Selection**: Have the developer select which bug they want to report work on, out of the open bugs assigned to them.
3. **Progress Reporting**: Collect descriptions of the work performed and insights gained.
4. **Status Updates**: Determine whether the bug is solved or not.

The conversation should have no more than 20 turns; beyond that, it should be ended by the bot.

#### Bug selection with function call
The entire bug database can't be part of the chatbot's context, the bugs assigned to the developer must be fetched through a function call, filtering by developer_id.

#### Structured Output
At the end of the conversation, a structured json output must be produced, with the following fields
- success (boolean): whether the conversation fulfilled its information gathering objectives
- report (optional, object)
  - bug_id
  - progress_note: timestamped data entry that should be added to the bug's progress_notes field
  - solved: whether the bug has been solved by the developer

Example:
```json
{
  "success": true,
  "report": {
    "bug_id": 1,
    "progress_note": "2024-01-15 14:30:00 - Updated regex pattern to handle special characters",
    "solved": true
  }
}
```

#### Technical Requirements
- Use Python as the primary language
- Write clean, well-documented code

#### Deliverables
1. **Python Project**: Complete implementation with your preferred project structure
2. **Documentation**: 
   - README with setup and usage instructions
   - Code comments explaining key decisions
3. **Results**: 
   - Examples of real conversation traces (either with a real human or AI simulated developer). Make sure you include a couple of adversarial and error cases.
   The traces should include messages, tool calls, and the final structured output.  
   **Examples of adversarial scenarios**: "developer tries to report on bugs not assigned to them", "developer gives vague or unhelpful responses".  
   **Examples of error cases**: Developer not found in dataset", "no bugs assigned to developer"


    - 2 performance evaluation metrics ran on the conversation examples (pick the performance evaluation metrics you see fit)

### Dataset Description
The system will work with two main datasets:

#### Developers Dataset
- `developer_id`: Unique identifier for each developer
- `name`: Developer's name

#### Bugs Dataset
- `bug_id`: Unique identifier for each bug
- `description`: Detailed description of the bug
- `assigned_dev`: ID of the developer assigned to the bug
- `status`: Current status (Open, In Progress, Testing, Resolved, Closed)
- `progress_notes`: Work that has been performed on the bug (timestamped entries)
- `solved`: Boolean indicating if the bug is solved

See [data readme for more details](./data/README.md)

### Example Conversations
The chatbot should guide developers through conversations like:

**Developer Identification:**
- Bot: "Hi! What's your name?"
- Dev: "I'm Alice Johnson"
- Bot: "Hello Alice! I can see you're assigned to several bugs. Which one would you like to report progress on?"

**Bug Selection:**
- Bot: "Here are your assigned bugs: Bug #1 - Authentication issue, Bug #5 - Email notifications. Which one?"
- Dev: "The authentication one"
- Bot: "Great! You're working on Bug #1 - User authentication fails when using special characters in password. What work have you done on this?"

**Progress Reporting:**
- Dev: "I investigated the password validation logic and found the regex pattern needs updating"
- Bot: "That's helpful! Any insights or specific changes you made?"
- Dev: "Yes, I updated the regex to handle special characters properly and tested it"
- Bot: "Excellent work! Is this bug now solved, or do you need to do more work?"

**Status Updates:**
- Dev: "I think it's solved, I tested it thoroughly"
- Bot: "Perfect! I'll mark Bug #1 as resolved and add your progress notes. Is there anything else you'd like to report?"

### Timeline
- **Submission**: Send a link to your GitHub repository or zip file
- **Presentation**: Be prepared to walk through your solution in 15 minutes

### Questions?
If you have any questions about the problem or requirements, please reach out to julien@quintess.ai.

Good luck!