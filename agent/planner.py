from typing import List, Dict, Any
import json
import re
import os

from llm.local_llm_client import LocalLLMClient
from core.indexer import CodeIndexer
from core.query_router import QueryRouter, QueryType
from tools.github_helper import clone_github_repo, get_repo_url_from_query


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
        
        Flow:
        1. Check if user wants to analyze a GitHub repo
        2. Classify query type
        3. If metadata: answer locally
        4. If search/reasoning: retrieve relevant chunks
        5. Ask LLM to plan given context
        """
        
        # Check for GitHub repo in query
        github_url = get_repo_url_from_query(user_query)
        if github_url:
            return self._plan_github_analysis(user_query, github_url)
        
        query_type = self.router.classify(user_query)
        print(f"[PLANNER] Query type: {query_type.value}")
        
        # METADATA: Handle locally
        if query_type == QueryType.METADATA:
            return self._handle_metadata_query(user_query)
        
        # TOOL: Direct execution
        if query_type == QueryType.TOOL_CALL:
            return self._extract_tool_calls(user_query)
        
        # SEARCH / REASONING: Retrieve + LLM
        retrieved_context = self._retrieve_context(user_query)
        return self._plan_with_llm(user_query, retrieved_context, query_type)
    
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
