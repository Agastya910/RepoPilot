"""
Edit Executor - Orchestrates the complete editing flow

This is the CORE of the new editing system.

Flow:
  1. Detect EDIT query
  2. Parse edit intent
  3. Retrieve full file
  4. Generate code patch
  5. Show diff
  6. Get user confirmation
  7. Apply edit with rollback support
  8. Log interaction for training

No ReAct loops. Direct path. Fast.
"""

import os
import difflib
from typing import Dict, Optional, Tuple
from agent.edit_detector import EditDetector
from agent.code_generator import CodeGenerator
from tools.file_retriever import FileRetriever
from core.safe_editor import SafeEditor
from core.interaction_logger import InteractionLogger


class EditExecutor:
    """
    Orchestrates editing queries.
    
    This is where the magic happens - converts user intent → actual code changes.
    """
    
    def __init__(self, repo_path: str, llm_client=None):
        self.repo_path = repo_path
        self.file_retriever = FileRetriever(repo_path)
        self.code_generator = CodeGenerator(llm_client)
        self.safe_editor = SafeEditor(repo_path)
        self.interaction_logger = InteractionLogger(repo_path)
    
    def execute_edit(self, query: str, auto_apply: bool = False) -> Dict:
        """
        Execute an edit query.
        
        Returns:
        {
            "success": bool,
            "file_path": str,
            "original_content": str,
            "modified_content": str,
            "diff": str,
            "applied": bool,
            "error": str or None,
            "interaction_id": str
        }
        """
        
        print("\n[EDIT PATH] Entering editing mode...")
        
        # Step 1: Detect and parse edit intent
        if not EditDetector.is_edit_query(query):
            return {
                "success": False,
                "error": "Not a valid edit query",
                "applied": False
            }
        
        intent = EditDetector.extract_edit_intent(query)
        print(f"[INTENT] Detected: {intent['operation'].value}")
        print(f"[TARGET] File: {intent['file_target']}")
        
        # Step 2: Find the target file
        file_path = self._locate_file(intent['file_target'])
        if not file_path:
            error = f"Could not find file: {intent['file_target']}"
            print(f"[ERROR] {error}")
            return {
                "success": False,
                "error": error,
                "applied": False
            }
        
        print(f"[RETRIEVED] {file_path}")
        
        # Step 3: Get current file content
        original_content = self.file_retriever.get_file(file_path)
        if not original_content:
            error = f"Could not read file: {file_path}"
            print(f"[ERROR] {error}")
            return {
                "success": False,
                "error": error,
                "applied": False
            }
        
        print(f"[SIZE] {len(original_content)} bytes, {len(original_content.split(chr(10)))} lines")
        
        # Step 4: Generate code modifications
        modified_content, gen_success, gen_error = self.code_generator.generate_patch(
            file_path=file_path,
            file_content=original_content,
            change_request=intent['change_description'],
            target_element=intent.get('target_element')
        )
        
        if not gen_success:
            print(f"[ERROR] Code generation failed: {gen_error}")
            return {
                "success": False,
                "error": gen_error,
                "applied": False,
                "file_path": file_path
            }
        
        # Step 5: Generate and show diff
        diff = self._generate_diff(original_content, modified_content, file_path)
        print("\n[DIFF]")
        print(diff)
        
        # Step 6: Get user confirmation (unless auto_apply)
        if not auto_apply:
            response = input("\n[CONFIRM] Apply this change? (y/n/review): ").strip().lower()
            
            if response == 'review':
                # Show more context
                print("\n[FULL MODIFIED FILE]")
                print("=" * 80)
                print(modified_content)
                print("=" * 80)
                response = input("\n[CONFIRM] Apply this change? (y/n): ").strip().lower()
            
            if response != 'y':
                print("[CANCELLED] Edit not applied")
                return {
                    "success": True,
                    "file_path": file_path,
                    "original_content": original_content,
                    "modified_content": modified_content,
                    "diff": diff,
                    "applied": False
                }
        
        # Step 7: Apply edit with SafeEditor (syntax checking + rollback)
        is_valid, proposal, error = self.safe_editor.propose_edit(
            file_path=file_path,
            proposed_content=modified_content,
            description=intent['change_description'],
            reasoning="Generated by CodeGenerator",
            affected_lines=(0, len(original_content.split('\n')))
        )
        
        if not is_valid:
            print(f"[SYNTAX ERROR] {error}")
            return {
                "success": False,
                "error": error,
                "file_path": file_path,
                "applied": False
            }
        
        # Apply with snapshot for rollback
        success, apply_error = self.safe_editor.apply_edit(
            proposal, 
            edit_id=self._gen_edit_id(),
            create_snapshot=True
        )
        
        if not success:
            print(f"[ERROR] Could not apply edit: {apply_error}")
            return {
                "success": False,
                "error": apply_error,
                "file_path": file_path,
                "applied": False
            }
        
        print(f"[SUCCESS] ✅ {file_path} updated")
        
        # Step 8: Log interaction for training data
        self._log_interaction(
            query=query,
            file_path=file_path,
            original_content=original_content,
            modified_content=modified_content,
            intent=intent,
            success=True
        )
        
        return {
            "success": True,
            "file_path": file_path,
            "original_content": original_content,
            "modified_content": modified_content,
            "diff": diff,
            "applied": True,
            "error": None
        }
    
    def _locate_file(self, target: Optional[str]) -> Optional[str]:
        """
        Find the file being edited.
        
        Strategies:
        1. If target is a full path, use it
        2. If target is a filename, search repo
        3. Use fuzzy matching
        """
        
        if not target:
            return None
        
        # Strategy 1: Direct path
        if self.file_retriever.get_file(target) is not None:
            return target
        
        # Strategy 2: Add .py if missing (common case)
        if not target.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs')):
            with_ext = target + '.py'
            if self.file_retriever.get_file(with_ext) is not None:
                return with_ext
        
        # Strategy 3: Search repo
        found = self.file_retriever.find_file_by_name(target)
        if found:
            return found
        
        # Strategy 4: Search by pattern
        results = self.file_retriever.find_files_by_pattern(f"*{target}*")
        if results:
            return results[0]  # Return first match
        
        return None
    
    def _generate_diff(self, original: str, modified: str, file_path: str) -> str:
        """
        Generate unified diff between original and modified.
        """
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        diff_lines = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        )
        
        diff_text = '\n'.join(diff_lines)
        return diff_text or "(No changes)"
    
    def _gen_edit_id(self) -> str:
        """Generate unique edit ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _log_interaction(self, 
                        query: str,
                        file_path: str,
                        original_content: str,
                        modified_content: str,
                        intent: Dict,
                        success: bool):
        """
        Log this interaction for training data collection.
        """
        
        try:
            self.interaction_logger.log_interaction(
                query=query,
                repo_name=os.path.basename(self.repo_path),
                query_type="editing",
                file_selected=file_path,
                code_change_proposed=self._get_diff_stat(original_content, modified_content),
                code_change_description=intent.get('change_description', ''),
                code_change_reasoning="Generated by CodeGenerator",
                syntax_valid=True,  # SafeEditor already validated
                user_approved=True,  # User confirmed
                success=success,
                tokens_used=0,  # Could track this
                latency_ms=0.0   # Could track this
            )
        except Exception as e:
            print(f"[LOGGING] Warning: could not log interaction: {e}")
    
    def _get_diff_stat(self, original: str, modified: str) -> str:
        """Get diff statistics for logging."""
        orig_lines = len(original.split('\n'))
        mod_lines = len(modified.split('\n'))
        delta = mod_lines - orig_lines
        return f"{orig_lines} → {mod_lines} lines ({delta:+d})"


class EditSession:
    """
    Maintains context across multiple edits in a session.
    
    Useful for: "First edit X, then edit Y"
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.executor = EditExecutor(repo_path)
        self.edits_made = []
    
    def execute_edits(self, edits: list) -> list:
        """
        Execute multiple edits in sequence.
        
        Args:
            edits: List of edit queries
        
        Returns:
            List of results
        """
        results = []
        for edit_query in edits:
            result = self.executor.execute_edit(edit_query)
            results.append(result)
            
            if result['applied']:
                self.edits_made.append({
                    'query': edit_query,
                    'file': result['file_path']
                })
            else:
                print(f"[SESSION] Edit skipped due to error: {result.get('error')}")
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary of edits made in this session."""
        return {
            "total_edits": len(self.edits_made),
            "files_modified": list(set(e['file'] for e in self.edits_made)),
            "edits": self.edits_made
        }
