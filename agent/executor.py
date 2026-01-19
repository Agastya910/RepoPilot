from typing import List, Dict, Any
import json
import os

from tools import repo_scanner, code_search, file_io, diff_writer, git_cloner
from tools.github_helper import clone_github_repo
from core.indexer import CodeIndexer


class Executor:
    """
    Executor: runs tool calls from the planner.
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        
        self.tools = {
            "scan_repo": repo_scanner.scan_repo,
            "search_code": code_search.search_code,
            "read_file": file_io.read_file,
            "write_diff": diff_writer.write_diff,
            "git_clone": git_cloner.clone_repo,
            "github_clone": self._github_clone_tool,
            "github_analyze": self._github_analyze_tool,
            "report": self._report_tool,
            "llm_analysis": self._llm_analysis_tool,
        }
    
    def _github_clone_tool(self, repo_url: str, dest_path: str, timeout: int = 120) -> Dict[str, Any]:
        """Clone a GitHub repo."""
        return clone_github_repo(repo_url, dest_path, timeout)
    
    def _github_analyze_tool(self, repo_path: str, query: str) -> str:
        """
        Analyze a cloned GitHub repo by re-indexing and answering the query.
        """
        if not os.path.exists(repo_path):
            return f"❌ Repository path does not exist: {repo_path}"
        
        try:
            print(f"[ANALYZER] Indexing cloned repository at {repo_path}...")
            indexer = CodeIndexer(repo_path)
            
            file_count = indexer.get_file_count()
            print(f"[ANALYZER] Indexed {file_count} files")
            
            # Search in the cloned repo
            print(f"[ANALYZER] Searching for: '{query}'")
            results = indexer.search(query, k=5)
            
            if not results:
                summary = indexer.get_architecture_summary()
                return f"Repository indexed ({file_count} files).\n\nArchitecture:\n{summary}\n\nNo specific matches found for '{query}', but repository is ready for analysis."
            
            # Format results
            analysis = f"Repository indexed ({file_count} files).\n\nRelevant findings for '{query}':\n\n"
            for i, r in enumerate(results, 1):
                analysis += f"{i}. **{r['file_path']}** ({r['language']})\n"
                analysis += f"   Lines {r['start_line']}-{r['end_line']}\n"
                analysis += f"   Relevance: {r['score']:.2f}\n"
                analysis += f"   Content:\n   ```\n{r['content'][:300]}\n   ```\n\n"
            
            return analysis
        
        except Exception as e:
            return f"❌ Error analyzing repository: {str(e)}"
    
    def _report_tool(self, message: str) -> str:
        """Simple message reporting tool."""
        return message
    
    def _llm_analysis_tool(self, query: str, analysis: str) -> str:
        """Store LLM analysis result."""
        return analysis
    
    def execute_plan(self, plan: List[Dict[str, Any]]) -> List[Any]:
        """Execute a plan of tool calls."""
        results = []
        
        for tool_call in plan:
            tool_name = tool_call.get("tool_name")
            args = tool_call.get("args", {})
            
            if tool_name not in self.tools:
                results.append({
                    "tool": tool_name,
                    "error": f"Unknown tool: {tool_name}"
                })
                continue
            
            print(f"[EXECUTOR] → {tool_name}")
            
            try:
                result = self.tools[tool_name](**args)
                results.append({"tool": tool_name, "result": result})
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "error": str(e)
                })
        
        return results
