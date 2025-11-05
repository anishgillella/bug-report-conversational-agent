"""
Data management module for loading and querying developers and bugs.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


class DataManager:
    """Manages loading and querying developer and bug data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataManager and load datasets.
        
        Args:
            data_dir: Directory containing bugs.json and developers.json
        """
        self.data_dir = Path(data_dir)
        self.developers = self._load_developers()
        self.bugs = self._load_bugs()
        self._build_indices()
    
    def _load_developers(self) -> List[Dict[str, Any]]:
        """Load developers from JSON file."""
        dev_file = self.data_dir / "developers.json"
        with open(dev_file, "r") as f:
            return json.load(f)
    
    def _load_bugs(self) -> List[Dict[str, Any]]:
        """Load bugs from JSON file."""
        bugs_file = self.data_dir / "bugs.json"
        with open(bugs_file, "r") as f:
            return json.load(f)
    
    def _build_indices(self):
        """Build lookup indices for faster queries."""
        # Developer ID -> Developer mapping
        self.dev_id_map = {dev["developer_id"]: dev for dev in self.developers}
        
        # Developer name -> Developer ID (case-insensitive)
        self.dev_name_map = {dev["name"].lower(): dev["developer_id"] for dev in self.developers}
        
        # Bug ID -> Bug mapping
        self.bug_id_map = {bug["bug_id"]: bug for bug in self.bugs}
    
    def find_developer_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a developer by name (case-insensitive).
        Supports exact matches and partial/fuzzy matches.
        
        Args:
            name: Developer name to search for
            
        Returns:
            Developer dict if found, None otherwise
        """
        name_lower = name.lower().strip()
        
        # Try exact match first
        dev_id = self.dev_name_map.get(name_lower)
        if dev_id:
            return self.dev_id_map[dev_id]
        
        # Try partial/fuzzy matching
        matches = self._find_partial_matches(name_lower)
        if len(matches) == 1:
            # Only one match - return it
            return matches[0]
        
        return None
    
    def _find_partial_matches(self, partial_name: str) -> List[Dict[str, Any]]:
        """
        Find developers by partial name match.
        Handles: first name, last name, or any word in the name.
        
        Args:
            partial_name: Partial name or first/last name only
            
        Returns:
            List of developer dicts that match
        """
        matches = []
        partial_lower = partial_name.lower().strip()
        
        for dev in self.developers:
            full_name = dev["name"].lower()
            name_parts = full_name.split()
            
            # Check if partial_name matches any part of the full name
            for part in name_parts:
                # Exact word match
                if part == partial_lower:
                    matches.append(dev)
                    break
                # Partial word match (e.g., "Ali" matches "Alice")
                elif part.startswith(partial_lower) and len(partial_lower) >= 3:
                    matches.append(dev)
                    break
        
        return matches
    
    def find_similar_developers(self, name: str) -> List[Dict[str, Any]]:
        """
        Find developers with similar names (for disambiguation).
        Used when user provides ambiguous input.
        
        Args:
            name: Name to search for
            
        Returns:
            List of similar developers
        """
        return self._find_partial_matches(name.lower().strip())
    
    def get_developer_by_id(self, dev_id: int) -> Optional[Dict[str, Any]]:
        """Get developer by ID."""
        return self.dev_id_map.get(dev_id)
    
    def get_bugs_for_developer(self, dev_id: int) -> List[Dict[str, Any]]:
        """
        Get all bugs assigned to a developer.
        This is the tool function called by the LLM.
        
        Args:
            dev_id: Developer ID
            
        Returns:
            List of bug dicts assigned to this developer
        """
        return [bug for bug in self.bugs if bug["assigned_dev"] == dev_id]
    
    def get_bug_by_id(self, bug_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific bug by ID."""
        return self.bug_id_map.get(bug_id)
    
    def list_all_developers(self) -> List[Dict[str, Any]]:
        """List all developers."""
        return self.developers
    
    def list_all_bugs(self) -> List[Dict[str, Any]]:
        """List all bugs."""
        return self.bugs
    
    def update_bug_progress(self, bug_id: int, progress_note: str, solved: bool) -> bool:
        """
        Update a bug with progress note and solved status.
        Saves changes to the bugs.json file.
        
        Args:
            bug_id: Bug ID to update
            progress_note: Progress note to add (with timestamp)
            solved: Whether the bug is solved
            
        Returns:
            True if successful, False otherwise
        """
        bug = self.bug_id_map.get(bug_id)
        if not bug:
            return False
        
        # Add progress note to existing notes
        if bug.get("progress_notes"):
            bug["progress_notes"] += f"\n{progress_note}"
        else:
            bug["progress_notes"] = progress_note
        
        # Update solved status
        bug["solved"] = solved
        
        # Save to file
        self._save_bugs()
        return True
    
    def _save_bugs(self) -> None:
        """Save bugs back to JSON file."""
        bugs_file = self.data_dir / "bugs.json"
        with open(bugs_file, "w") as f:
            json.dump(self.bugs, f, indent=2)
