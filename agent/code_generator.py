"""
Code Generator - Direct Code Synthesis

NOT an analyzer. NOT a reasoner.
PURELY generates code modifications based on requirements.

Think of it like: Give me the file + what to change → Get back modified file
"""

import re
from typing import Tuple, Optional
from llm.local_llm_client import LocalLLMClient
from core.streaming_executor import SyncStreamingExecutor


class CodeGenerator:
    """
    Generates code changes directly from requirements.
    
    Key principle: Don't ask the LLM to "analyze and explain"
    Ask it to "generate the modified code"
    
    This is fundamentally different from the ReAct analysis path.
    """
    
    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.streaming_executor = SyncStreamingExecutor(self.llm_client)
    
    def generate_patch(self,
                      file_path: str,
                      file_content: str,
                      change_request: str,
                      target_element: Optional[str] = None) -> Tuple[str, bool, Optional[str]]:
        """
        Generate modified file content.
        
        Args:
            file_path: Path to file being edited (e.g., "repo_scanner.py")
            file_content: Current complete file content
            change_request: What to change (e.g., "add max_depth parameter to scan_repo")
            target_element: Optional - function/class being modified
        
        Returns:
            (modified_content, success, error_message)
        """
        
        # Determine language for syntax highlighting in prompt
        language = self._get_language(file_path)
        
        # Build prompt - THIS IS DIFFERENT FROM ANALYSIS PROMPTS
        prompt = self._build_generation_prompt(
            file_path=file_path,
            file_content=file_content,
            change_request=change_request,
            target_element=target_element,
            language=language
        )
        
        # Call LLM with streaming
        print("[GENERATING]... ", end="", flush=True)
        response, success = self.streaming_executor.stream_llm_analysis_to_stdout(prompt)
        print()  # newline after streaming
        
        if not success:
            # Fallback to non-streaming
            response = self.llm_client.generate_text(prompt)
        
        if not response:
            return None, False, "LLM returned empty response"
        
        # Extract code from response
        modified_code = self._extract_code_from_response(response, language)
        
        if not modified_code:
            return None, False, "Could not extract code from LLM response"
        
        return modified_code, True, None
    
    def _build_generation_prompt(self,
                                file_path: str,
                                file_content: str,
                                change_request: str,
                                target_element: Optional[str],
                                language: str) -> str:
        """
        Build the prompt for code generation.
        
        THIS IS DIFFERENT FROM ANALYSIS PROMPTS!
        - Don't ask for explanation
        - Don't ask for reasoning
        - Ask for CODE
        """
        
        # Highlight target section if provided
        highlighted_content = file_content
        if target_element:
            highlighted_content = self._highlight_target(
                file_content, target_element, language
            )
        
        prompt = f"""You are a code editor. Your ONLY job is to generate code.

FILE: {file_path}
LANGUAGE: {language}

CURRENT CODE:
```{language}
{highlighted_content}
```

REQUIRED CHANGE:
{change_request}

INSTRUCTIONS:
1. Return ONLY the modified complete file
2. Keep the same coding style and structure
3. Preserve comments and docstrings
4. No explanations, no reasoning, just code
5. Return valid {language} syntax

MODIFIED CODE:
```{language}
"""
        
        return prompt
    
    def _highlight_target(self, file_content: str, target: str, language: str) -> str:
        """
        Highlight the target function/class in the file.
        
        Helps LLM focus on what to change.
        """
        lines = file_content.split('\n')
        result = []
        in_target = False
        
        for i, line in enumerate(lines):
            # Simple heuristic: look for def/class followed by target name
            if f"def {target}" in line or f"class {target}" in line:
                in_target = True
                result.append(f">>> {line}")  # Mark start
            elif in_target and (line.startswith('def ') or line.startswith('class ')):
                in_target = False
                result.append(f"<<< {line}")  # Mark end of previous
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _extract_code_from_response(self, response: str, language: str) -> Optional[str]:
        """
        Extract code block from LLM response.
        
        Handles formats like:
        ```python
        code here
        ```
        """
        
        # Look for code block with language
        pattern = rf"```{language}\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Fallback: just look for ``` blocks
        pattern = r"```\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Last resort: if response looks like code, return it
        if language == "python" and "def " in response or "class " in response:
            return response.strip()
        
        return None
    
    def _get_language(self, file_path: str) -> str:
        """Determine programming language from file extension."""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
        }
        
        for ext, lang in ext_to_lang.items():
            if file_path.endswith(ext):
                return lang
        
        return 'python'  # default
    
    def generate_with_context_window(self,
                                     file_content: str,
                                     change_request: str,
                                     context_size: int = 4000) -> Tuple[str, bool, Optional[str]]:
        """
        Generate code with explicit context window size.
        
        Useful for large files - you can slice context.
        """
        
        # For now, just use full file
        # In future, can implement smart context windowing
        return self._build_generation_prompt(
            file_path="<file>",
            file_content=file_content,
            change_request=change_request,
            target_element=None,
            language="python"
        )
    
    def batch_generate(self, file_changes: list) -> list:
        """
        Generate multiple file changes.
        
        Args:
            file_changes: List of {
                "file_path": str,
                "content": str,
                "request": str
            }
        
        Returns:
            List of (modified_content, success, error)
        """
        results = []
        for change in file_changes:
            result = self.generate_patch(
                file_path=change["file_path"],
                file_content=change["content"],
                change_request=change["request"]
            )
            results.append(result)
        return results
