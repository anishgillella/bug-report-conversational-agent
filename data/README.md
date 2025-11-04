# Data Folder

This folder contains the input data for the Quintess AI Engineer take-home test.

## Files Description

### `developers.json`
Dataset containing developer information.

**Columns:**
- `developer_id`: Unique identifier for each developer
- `name`: Developer's name

### `bugs.json`
Main dataset containing bug information for the chatbot system.

**Columns:**
- `bug_id`: Unique identifier for each bug
- `description`: Detailed description of the bug
- `assigned_dev`: ID of the developer assigned to the bug (references developers.developer_id)
- `status`: Current status of the bug (Open, In Progress, Testing, Resolved, Closed)
- `progress_notes`: Work that has been performed on the bug (timestamped entries, one per line)
- `solved`: Boolean indicating if the bug is solved (True/False)

**Progress Notes Format:**
Each entry in `progress_notes` follows the format: `YYYY-MM-DD HH:MM:SS - Description of work performed`
Entries are separated by newline characters (`\n`). This field can contain data even for open bugs, representing ongoing work.

## Data Usage Notes

- The datasets are related through the `assigned_dev` field in bugs.json
- Progress notes can contain timestamped work entries, even for open bugs
- You are free to modify, extend, or replace this data as needed for your solution, as long as you respect the overall structure and relationships between the datasets. Feel free to add more developers, bugs that might be useful for your chatbot implementation.