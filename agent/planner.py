from typing import List, Dict, Any
import json
import re
import os

from llm.local_llm_client import LocalLLMClient
from core.indexer import CodeIndexer
from core.query_router import QueryRouter, QueryType
from tools.github_helper import clone_github_repo, get_repo_url_from_query
from agent.reactor_framework import ReActRunner


class Planner:
    """
    Planner agent: decomposes user queries into tool calls.
    Now retrieval-aware: only sends relevant context to LLM.
    Supports GitHub repo analysis.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.llm_client = LocalLLMClient()
        self.indexer = CodeIndexer(repo_path)
        self.router = QueryRouter()
    
    def create_plan(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Create a plan for the user query.
        For obvious single-file edit requests, use a direct edit plan.
        Otherwise, use ReAct + fallback LLM analysis.
        """
        # 0) Simple heuristic: detect "edit <file>" pattern
        edit_plan = self._maybe_plan_edit(user_query)
        if edit_plan is not None:
            return edit_plan

        # 1) Use ReAct to reason (analysis / Q&A)
        react = ReActRunner(self.llm_client, self.repo_path)
        chunks = self._retrieve_context(user_query)
        initial_context = {
            "file_count": len(chunks),
            "retrieved_chunks": chunks,
        }
        result, trace = react.execute(user_query, initial_context)
        self.trace = trace

        # 2) If ReAct produced a string, report it
        if isinstance(result, str):
            return [{
                "tool_name": "report",
                "args": {"message": result},
            }]

        # 3) If ReAct produced a code-search-like list, serialize
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return [{
                "tool_name": "report",
                "args": {"message": json.dumps(result, indent=2)},
            }]

        # 4) Fallback to old LLM analysis
        query_type = self.router.classify(user_query)
        return self._plan_with_llm(user_query, chunks, query_type)
    
    def _maybe_plan_edit(self, user_query: str) -> List[Dict[str, Any]] | None:
        """
        Detect simple single-file edit requests and create a direct edit plan.

        Example:
        "can you edit the repo_scanner.py to make it scan one more layer..."
        """
        lower = user_query.lower()
        # crude but effective
        if "edit" not in lower and "change" not in lower and "modify" not in lower:
            return None

        # Try to extract a filename with .py
        match = re.search(r"\b([a-zA-Z0-9_]+\.py)\b", user_query)
        if not match:
            return None

        file_name = match.group(1)  # e.g. repo_scanner.py
        file_path = file_name  # tools/ is handled by indexer/context

        # Retrieve the most relevant chunks for that file
        chunks = self.indexer.search(file_name, k=3)
        context_str = "\n".join([
            f"File: {c['file_path']}\n```{c['language']}\n{c['content'][:500]}\n```"
            for c in chunks
        ])

        prompt = f"""
You are editing a Python utility that scans a repository.

User request:
\"\"\"{user_query}\"\"\"\n
Current implementation (context from the codebase):
{context_str if context_str else "[No context found, but file exists in tools/ directory]"}
    
Task:
1. Update {file_name} so that:
   - It can scan one more level of depth in subdirectories, OR
   - It allows the caller to specify a maximum depth for scanning.
2. Keep the public API compatible where possible.
3. Follow the existing coding style.
4. Return the full updated content of {file_name}, not a diff.

Respond in this exact JSON format:

{{
  "file_path": "tools/{file_name}",
  "description": "What you changed, in 1-2 sentences.",
  "reasoning": "Brief technical reasoning for the change.",
  "proposed_content": "FULL updated file content here."
}}
"""

        response = self.llm_client.generate_text(prompt)

        # Extract JSON
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON in LLM response")
            edit_data = json.loads(json_match.group())
        except Exception:
            # Fallback: just report the raw response
            return [{
                "tool_name": "report",
                "args": {"message": response},
            }]

        # Plan: just report the JSON to the CLI for now
        # (SafeEditor integration can come next)
        pretty = json.dumps(edit_data, indent=2)
        return [{
            "tool_name": "report",
            "args": {"message": pretty},
        }]

    def _plan_github_analysis(self, query: str, github_url: str) -> List[Dict[str, Any]]:
        """Plan for analyzing a GitHub repository."""
        print(f"[PLANNER] GitHub URL detected: {github_url}")
        print("[PLANNER] Planning: Clone → Index → Analyze")
        
        return [
            {
                "tool_name": "github_clone",
                "args": {
                    "repo_url": github_url,
                    "dest_path": "./analyzed_repo",
                    "timeout": 120
                }
            },
            {
                "tool_name": "github_analyze",
                "args": {
                    "repo_path": "./analyzed_repo",
                    "query": query
                }
            }
        ]
    
    def _handle_metadata_query(self, query: str) -> List[Dict[str, Any]]:
        """Answer metadata queries without LLM."""
        print("[PLANNER] Answering metadata query locally...")
        
        files = self.indexer.get_file_list()
        
        if "how many" in query.lower() and "files" in query.lower():
            return [{
                "tool_name": "report",
                "args": {"message": f"This repo has {len(files)} indexed files."}
            }]
        
        if "list" in query.lower():
            file_list = "\n".join([f"  - {f['path']} ({f['language']})" for f in files[:20]])
            if len(files) > 20:
                file_list += f"\n  ... and {len(files) - 20} more files"
            return [{
                "tool_name": "report",
                "args": {"message": f"Files in repo ({len(files)} total):\n{file_list}"}
            }]
        
        if "architecture" in query.lower() or "structure" in query.lower():
            return [{
                "tool_name": "report",
                "args": {"message": self.indexer.get_architecture_summary()}
            }]
        
        # Fallback
        return [{
            "tool_name": "report",
            "args": {"message": f"Found {len(files)} files in repository."}
        }]
    
    def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-5 relevant code chunks."""
        return self.indexer.search(query, k=5)
    
    def _plan_with_llm(
        self,
        user_query: str,
        retrieved_context: List[Dict[str, Any]],
        query_type: QueryType
    ) -> List[Dict[str, Any]]:
        """Use LLM to create plan given retrieved context."""
        
        context_str = "\n".join([
            f"File: {c['file_path']}\n```{c['language']}\n{c['content'][:500]}\n```"
            for c in retrieved_context
        ])
        
        prompt = f"""
You are a senior software engineer analyzing a codebase.

User Query: "{user_query}"

Relevant Code Context (top 5 chunks):
{context_str if context_str else "[No relevant code found]"}

Based on the query and context, provide a technical analysis. Be specific and reference file names.
Keep your response clear, concise, and actionable.
"""
        
        response = self.llm_client.generate_text(prompt)
        
        return [{
            "tool_name": "llm_analysis",
            "args": {
                "query": user_query,
                "analysis": response
            }
        }]
    
    def _extract_tool_calls(self, query: str) -> List[Dict[str, Any]]:
        """Extract git_clone or direct tool calls."""
        github_url = get_repo_url_from_query(query)
        if github_url:
            return [{
                "tool_name": "github_clone",
                "args": {
                    "repo_url": github_url,
                    "dest_path": "./analyzed_repo",
                    "timeout": 120
                }
            }]
        return []
