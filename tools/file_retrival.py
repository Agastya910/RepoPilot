"""
File Retriever - Load Complete Files for Editing Context

Unlike the semantic indexer which chunks files,
this loads ENTIRE files because editing needs full context.
"""

import os
from typing import Optional, Dict, List
from pathlib import Path


class FileRetriever:
    """
    Retrieves complete file content for editing operations.
    
    Unlike Indexer which chunks files semantically,
    this is FULL FILE retrieval for context preservation.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
    
    def get_file(self, file_path: str) -> Optional[str]:
        """
        Get complete file content.
        
        Args:
            file_path: Relative path to file (e.g., "tools/repo_scanner.py")
        
        Returns:
            Complete file content or None if not found
        """
        
        abs_path = os.path.join(self.repo_path, file_path)
        
        # Safety: prevent path traversal
        abs_path = os.path.abspath(abs_path)
        repo_abs = os.path.abspath(self.repo_path)
        
        if not abs_path.startswith(repo_abs):
            return None
        
        if not os.path.exists(abs_path):
            return None
        
        if not os.path.isfile(abs_path):
            return None
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[FILE RETRIEVER] Error reading {file_path}: {e}")
            return None
    
    def get_file_with_line_numbers(self, file_path: str) -> Optional[List[str]]:
        """
        Get file content with line numbers for reference.
        
        Returns:
            List of "000: code line" strings or None
        """
        content = self.get_file(file_path)
        if not content:
            return None
        
        lines = content.split('\n')
        numbered = [f"{i+1:04d}: {line}" for i, line in enumerate(lines)]
        return numbered
    
    def get_function(self, file_path: str, function_name: str) -> Optional[Dict]:
        """
        Extract a specific function from a file.
        
        Returns:
        {
            "name": str,
            "start_line": int,
            "end_line": int,
            "content": str,
            "signature": str
        }
        """
        content = self.get_file(file_path)
        if not content:
            return None
        
        lines = content.split('\n')
        
        # Find function definition
        func_pattern = f"def {function_name}"
        start_line = None
        
        for i, line in enumerate(lines):
            if func_pattern in line:
                start_line = i
                break
        
        if start_line is None:
            return None
        
        # Find end of function (next def/class at same or lesser indentation)
        start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = len(lines)
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            
            line_indent = len(line) - len(line.lstrip())
            
            if (line.startswith('def ') or line.startswith('class ')) and line_indent <= start_indent:
                end_line = i
                break
        
        func_lines = lines[start_line:end_line]
        
        return {
            "name": function_name,
            "start_line": start_line,
            "end_line": end_line,
            "content": '\n'.join(func_lines),
            "signature": func_lines[0] if func_lines else ""
        }
    
    def get_class(self, file_path: str, class_name: str) -> Optional[Dict]:
        """
        Extract a specific class from a file.
        
        Returns:
        {
            "name": str,
            "start_line": int,
            "end_line": int,
            "content": str,
            "methods": List[str]
        }
        """
        content = self.get_file(file_path)
        if not content:
            return None
        
        lines = content.split('\n')
        
        # Find class definition
        class_pattern = f"class {class_name}"
        start_line = None
        
        for i, line in enumerate(lines):
            if class_pattern in line:
                start_line = i
                break
        
        if start_line is None:
            return None
        
        # Find end of class
        start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = len(lines)
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            
            line_indent = len(line) - len(line.lstrip())
            
            if line.startswith('class ') and line_indent <= start_indent:
                end_line = i
                break
        
        class_lines = lines[start_line:end_line]
        
        # Extract methods
        methods = []
        for line in class_lines:
            if line.strip().startswith('def '):
                method_name = line.strip().split('(')[0].replace('def ', '')
                methods.append(method_name)
        
        return {
            "name": class_name,
            "start_line": start_line,
            "end_line": end_line,
            "content": '\n'.join(class_lines),
            "methods": methods
        }
    
    def find_file_by_name(self, filename: str) -> Optional[str]:
        """
        Search repo for a file by name (not full path).
        
        Returns:
            Relative path to file or None
        """
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if file == filename:
                    abs_path = os.path.join(root, file)
                    return os.path.relpath(abs_path, self.repo_path)
        return None
    
    def find_files_by_pattern(self, pattern: str) -> List[str]:
        """
        Find files matching a pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "*scanner*")
        
        Returns:
            List of relative file paths
        """
        from fnmatch import fnmatch
        
        matches = []
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if fnmatch(file, pattern):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, self.repo_path)
                    matches.append(rel_path)
        
        return matches
    
    def list_directory(self, dir_path: str = "") -> Dict:
        """
        List contents of a directory.
        
        Returns:
        {
            "directories": [],
            "files": [],
            "total_size": int
        }
        """
        abs_path = os.path.join(self.repo_path, dir_path)
        
        if not os.path.exists(abs_path):
            return {"error": "Directory not found"}
        
        try:
            contents = os.listdir(abs_path)
            dirs = []
            files = []
            total_size = 0
            
            for item in contents:
                item_path = os.path.join(abs_path, item)
                
                if os.path.isdir(item_path):
                    dirs.append(item)
                elif os.path.isfile(item_path):
                    files.append(item)
                    total_size += os.path.getsize(item_path)
            
            return {
                "directories": sorted(dirs),
                "files": sorted(files),
                "total_size": total_size
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get metadata about a file.
        
        Returns:
        {
            "path": str,
            "size": int,
            "lines": int,
            "language": str,
            "has_syntax_errors": bool
        }
        """
        abs_path = os.path.join(self.repo_path, file_path)
        
        if not os.path.exists(abs_path):
            return None
        
        content = self.get_file(file_path)
        if not content:
            return None
        
        # Determine language
        _, ext = os.path.splitext(file_path)
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JSX',
            '.tsx': 'TSX',
            '.go': 'Go',
            '.rs': 'Rust',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
        }
        language = ext_to_lang.get(ext, 'Unknown')
        
        return {
            "path": file_path,
            "size": len(content),
            "lines": len(content.split('\n')),
            "language": language,
            "has_syntax_errors": False  # Could add syntax checking here
        }
