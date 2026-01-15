"""
Safe Code Editor with verification, version control, and user confirmation.

Key features:
- Syntax validation (AST-based for Python)
- Diff generation and display
- User confirmation flow before applying changes
- Snapshot/rollback capability
- Pre-flight checks (imports, dependencies)
"""

import ast
import difflib
import os
import json
import hashlib
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil


@dataclass
class CodeSnapshot:
    """A snapshot of file state at a point in time."""
    file_path: str
    content: str
    timestamp: str
    checksum: str
    edit_id: str


@dataclass
class EditProposal:
    """Proposed code edit with metadata."""
    file_path: str
    original_content: str
    proposed_content: str
    description: str
    reasoning: str  # LLM reasoning for the edit
    risk_level: str  # "low", "medium", "high"
    affected_lines: Tuple[int, int]  # start, end


class SyntaxValidator:
    """Validates code syntax for various languages."""
    
    @staticmethod
    def validate_python(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax using AST.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def validate_javascript(code: str) -> Tuple[bool, Optional[str]]:
        """
        Basic JS validation (checks for common syntax issues).
        Full validation requires a JS parser.
        """
        # Simple heuristics for now
        open_braces = code.count("{")
        close_braces = code.count("}")
        
        if open_braces != close_braces:
            return False, f"Mismatched braces: {open_braces} opening, {close_braces} closing"
        
        return True, None
    
    @staticmethod
    def validate_by_extension(file_path: str, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".py":
            return SyntaxValidator.validate_python(code)
        elif ext in [".js", ".jsx", ".ts", ".tsx"]:
            return SyntaxValidator.validate_javascript(code)
        else:
            # Unknown language - skip validation
            return True, None


class DiffGenerator:
    """Generate and format diffs for code changes."""
    
    @staticmethod
    def generate_unified_diff(
        original: str,
        modified: str,
        file_path: str,
        context_lines: int = 3
    ) -> str:
        """
        Generate unified diff format.
        
        Args:
            original: Original content
            modified: Modified content
            file_path: Path to file (for header)
            context_lines: Lines of context around changes
            
        Returns:
            Unified diff string
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"{file_path} (original)",
            tofile=f"{file_path} (modified)",
            n=context_lines
        )
        
        return "".join(diff)
    
    @staticmethod
    def display_diff(diff_str: str):
        """Display diff with colors for terminal."""
        for line in diff_str.splitlines():
            if line.startswith("---") or line.startswith("+++"):
                print(f"\033[94m{line}\033[0m")  # Blue headers
            elif line.startswith("-"):
                print(f"\033[91m{line}\033[0m")  # Red removals
            elif line.startswith("+"):
                print(f"\033[92m{line}\033[0m")  # Green additions
            elif line.startswith("@@"):
                print(f"\033[93m{line}\033[0m")  # Yellow markers
            else:
                print(line)


class VersionControl:
    """Simple file-based version control and rollback."""
    
    def __init__(self, repo_path: str, snapshots_dir: str = ".repopilot/snapshots"):
        self.repo_path = repo_path
        self.snapshots_dir = os.path.join(repo_path, snapshots_dir)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
    def create_snapshot(self, file_path: str, edit_id: str) -> CodeSnapshot:
        """Create a snapshot of current file state."""
        abs_path = os.path.join(self.repo_path, file_path)
        
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        checksum = hashlib.sha256(content.encode()).hexdigest()
        
        snapshot = CodeSnapshot(
            file_path=file_path,
            content=content,
            timestamp=datetime.now().isoformat(),
            checksum=checksum,
            edit_id=edit_id
        )
        
        # Save snapshot
        snapshot_path = os.path.join(
            self.snapshots_dir,
            f"{edit_id}_{hashlib.md5(file_path.encode()).hexdigest()}.json"
        )
        
        with open(snapshot_path, "w") as f:
            json.dump(asdict(snapshot), f, indent=2)
        
        return snapshot
    
    def restore_snapshot(self, edit_id: str, file_path: str) -> bool:
        """Restore file from snapshot."""
        # Find snapshot file
        snapshot_files = [
            f for f in os.listdir(self.snapshots_dir)
            if f.startswith(edit_id)
        ]
        
        if not snapshot_files:
            return False
        
        with open(os.path.join(self.snapshots_dir, snapshot_files[0])) as f:
            snapshot_data = json.load(f)
        
        # Write back to file
        abs_path = os.path.join(self.repo_path, file_path)
        with open(abs_path, "w") as f:
            f.write(snapshot_data["content"])
        
        return True
    
    def get_history(self, file_path: str) -> List[CodeSnapshot]:
        """Get edit history for a file."""
        history = []
        
        for snapshot_file in os.listdir(self.snapshots_dir):
            with open(os.path.join(self.snapshots_dir, snapshot_file)) as f:
                snap_data = json.load(f)
            
            if snap_data["file_path"] == file_path:
                snapshot = CodeSnapshot(**snap_data)
                history.append(snapshot)
        
        return sorted(history, key=lambda s: s.timestamp)


class SafeEditor:
    """Main editor class combining validation, diff, and version control."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.version_control = VersionControl(repo_path)
        self.validator = SyntaxValidator()
        self.diff_gen = DiffGenerator()
        
    def propose_edit(
        self,
        file_path: str,
        proposed_content: str,
        description: str,
        reasoning: str,
        affected_lines: Tuple[int, int] = (0, 0)
    ) -> Tuple[bool, EditProposal, Optional[str]]:
        """
        Propose a code edit with validation.
        
        Returns:
            (is_valid, proposal_or_error_detail, error_message)
        """
        abs_path = os.path.join(self.repo_path, file_path)
        
        # Read original
        if not os.path.exists(abs_path):
            return False, None, f"File not found: {file_path}"
        
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            original_content = f.read()
        
        # Validate syntax
        is_valid, error = self.validator.validate_by_extension(
            file_path,
            proposed_content
        )
        
        if not is_valid:
            return False, None, f"Syntax error: {error}"
        
        # Assess risk level
        risk = self._assess_risk(original_content, proposed_content, file_path)
        
        proposal = EditProposal(
            file_path=file_path,
            original_content=original_content,
            proposed_content=proposed_content,
            description=description,
            reasoning=reasoning,
            risk_level=risk,
            affected_lines=affected_lines
        )
        
        return True, proposal, None
    
    def _assess_risk(self, original: str, modified: str, file_path: str) -> str:
        """Assess risk level of change."""
        # Count changed lines
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        diff = list(difflib.unified_diff(original_lines, modified_lines, n=0))
        changed_lines = len([l for l in diff if l.startswith("+") or l.startswith("-")])
        
        # Check if touching imports/critical sections
        is_import_heavy = "import" in modified
        
        if changed_lines <= 5 and not is_import_heavy:
            return "low"
        elif changed_lines <= 20:
            return "medium"
        else:
            return "high"
    
    def display_proposal(self, proposal: EditProposal):
        """Display edit proposal to user."""
        print("\n" + "="*70)
        print("📝 EDIT PROPOSAL")
        print("="*70)
        print(f"File: {proposal.file_path}")
        print(f"Risk Level: {proposal.risk_level.upper()}")
        print(f"\nDescription: {proposal.description}")
        print(f"\nReasoning:\n{proposal.reasoning}")
        print("\n" + "-"*70)
        print("DIFF:")
        print("-"*70)
        
        diff = self.diff_gen.generate_unified_diff(
            proposal.original_content,
            proposal.proposed_content,
            proposal.file_path
        )
        self.diff_gen.display_diff(diff)
        
        print("="*70 + "\n")
    
    def apply_edit(
        self,
        proposal: EditProposal,
        edit_id: str,
        create_snapshot: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply an edit to file.
        
        Args:
            proposal: EditProposal object
            edit_id: Unique edit identifier
            create_snapshot: Whether to create pre-edit snapshot
            
        Returns:
            (success, error_message)
        """
        abs_path = os.path.join(self.repo_path, proposal.file_path)
        
        try:
            # Create pre-edit snapshot
            if create_snapshot:
                self.version_control.create_snapshot(proposal.file_path, edit_id)
            
            # Apply change
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(proposal.proposed_content)
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def ask_user_confirmation(
        self,
        proposal: EditProposal
    ) -> bool:
        """Ask user for confirmation before applying edit."""
        self.display_proposal(proposal)
        
        print("\n" + "🤔 Apply this edit?" + "\n")
        
        while True:
            response = input("(y)es, (n)o, (d)iff, (c)ancel > ").strip().lower()
            
            if response == "y":
                return True
            elif response == "n":
                print("❌ Edit rejected.")
                return False
            elif response == "d":
                diff = self.diff_gen.generate_unified_diff(
                    proposal.original_content,
                    proposal.proposed_content,
                    proposal.file_path
                )
                self.diff_gen.display_diff(diff)
                print()
            elif response == "c":
                print("⚠️ Cancelled.")
                return False
            else:
                print("Invalid response. Try again.")
    
    def rollback_to_snapshot(self, edit_id: str, file_path: str) -> bool:
        """Rollback to previous snapshot."""
        return self.version_control.restore_snapshot(edit_id, file_path)
