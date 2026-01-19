"""
Edit Query Detector

Identifies when a query is requesting code changes vs analysis.
This is THE gateway to the editing path - everything starts here.
"""

import re
from typing import Dict, Optional
from enum import Enum


class EditOperationType(Enum):
    """Types of edit operations."""
    MODIFY_FUNCTION = "modify_function"
    MODIFY_FILE = "modify_file"
    ADD_FUNCTION = "add_function"
    ADD_FEATURE = "add_feature"
    FIX_BUG = "fix_bug"
    REFACTOR = "refactor"
    UNKNOWN = "unknown"


class EditDetector:
    """
    Detects if a query is requesting code changes.
    
    Key insight: Different from analysis queries!
    - Analysis: "explain X" → return text
    - Edit: "change X to Y" → return code
    """
    
    # HIGH CONFIDENCE edit keywords
    EDIT_KEYWORDS = {
        "edit", "change", "modify", "fix", "refactor",
        "rewrite", "improve", "update", "enhance",
        "replace", "remove", "delete", "add",
        "implement", "create", "write", "convert"
    }
    
    # Context keywords - what are we editing?
    TARGET_KEYWORDS = {
        "function", "class", "method", "file", "code",
        "parameter", "argument", "logic", "implementation",
        "handler", "module", "component"
    }
    
    # Patterns for specific operations
    PATTERNS = {
        "add_parameter": r"add\s+(?:parameter|param|arg|argument).*named?.*\b(\w+)\b",
        "remove_parameter": r"remove\s+(?:parameter|param|arg).*\b(\w+)\b",
        "rename": r"rename\s+(\w+)\s+to\s+(\w+)",
        "add_error_handling": r"(?:add|implement).*(?:error|exception|try|catch)",
        "add_validation": r"(?:add|implement).*(?:validation|check|verify)",
        "improve_performance": r"(?:improve|optimize|speed.*up)",
        "fix_bug": r"(?:fix|correct|solve|resolve).*(?:bug|issue|problem)",
    }
    
    @staticmethod
    def is_edit_query(query: str) -> bool:
        """
        Determine if query is requesting code changes.
        
        Returns: True if edit query, False if analysis/explanation
        """
        query_lower = query.lower()
        
        # Strong signals: edit verb + target
        has_edit_keyword = any(
            f" {kw} " in f" {query_lower} "  # Word boundaries
            for kw in EditDetector.EDIT_KEYWORDS
        )
        
        if not has_edit_keyword:
            return False
        
        # Make sure it's not just asking about capability
        # "can you edit X?" vs "edit X to Y"
        if query_lower.startswith(("can you", "would you", "could you", "is it able")):
            # These are usually capability questions, not change requests
            return False
        
        return True
    
    @staticmethod
    def extract_edit_intent(query: str) -> Dict[str, str]:
        """
        Parse edit query to extract structured information.
        
        Returns:
        {
            "is_edit": bool,
            "file_target": str or None,
            "operation": EditOperationType,
            "change_description": str,
            "confidence": float
        }
        """
        query_lower = query.lower()
        
        if not EditDetector.is_edit_query(query):
            return {
                "is_edit": False,
                "file_target": None,
                "operation": EditOperationType.UNKNOWN,
                "change_description": query,
                "confidence": 0.0
            }
        
        # Extract file/function target
        # Patterns: "edit repo_scanner", "change scan_repo function", "modify file X"
        file_target = None
        
        # Look for file names (*.py, *.js, etc.)
        file_match = re.search(r'(?:file\s+)?(\w+\.(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c))', query_lower)
        if file_match:
            file_target = file_match.group(1)
        
        # Or just the word after edit/change/modify
        if not file_target:
            for kw in ["edit", "change", "modify", "fix", "refactor"]:
                match = re.search(rf'{kw}\s+(?:the\s+)?(?:function\s+)?(\w+)', query_lower)
                if match:
                    file_target = match.group(1)
                    break
        
        # Determine operation type
        operation = EditDetector._classify_operation(query_lower)
        
        # Extract description of what to change
        change_description = query
        for kw in ["to ", "add ", "fix ", "make ", "by "]:
            if kw in query_lower:
                idx = query_lower.find(kw)
                change_description = query[idx:]
                break
        
        return {
            "is_edit": True,
            "file_target": file_target,
            "operation": operation,
            "change_description": change_description,
            "confidence": 0.8 if file_target else 0.6
        }
    
    @staticmethod
    def _classify_operation(query_lower: str) -> EditOperationType:
        """Classify the type of edit being requested."""
        
        if "add" in query_lower and "parameter" in query_lower:
            return EditOperationType.ADD_FUNCTION
        elif "remove" in query_lower or "delete" in query_lower:
            return EditOperationType.MODIFY_FUNCTION
        elif "fix" in query_lower or "bug" in query_lower:
            return EditOperationType.FIX_BUG
        elif "refactor" in query_lower:
            return EditOperationType.REFACTOR
        elif "improve" in query_lower or "optimize" in query_lower:
            return EditOperationType.IMPROVE
        elif "add" in query_lower and ("feature" in query_lower or "functionality" in query_lower):
            return EditOperationType.ADD_FEATURE
        else:
            return EditOperationType.MODIFY_FUNCTION
    
    @staticmethod
    def should_use_edit_path(query: str) -> bool:
        """
        Final decision: Should this query use the EDIT path instead of ANALYSIS path?
        
        This is the critical gateway.
        """
        return EditDetector.is_edit_query(query)
