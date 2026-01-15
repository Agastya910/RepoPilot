"""
ReAct (Reasoning + Acting) Framework Implementation.

Follows the pattern: Observe → Thought → Action → Observe → ...

Research: https://arxiv.org/abs/2210.03629
This enables explicit reasoning traces visible to user and better decision-making.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from llm.local_llm_client import LocalLLMClient
from core.streaming_executor import SyncStreamingExecutor
from tools import code_search
from tools import file_io
from core.safe_editor import SafeEditor
import os


class ActionType(Enum):
    """Types of actions available in the ReAct loop."""
    SEARCH = "search"  # Search code index
    RETRIEVE = "retrieve"  # Get file content
    EDIT = "edit"  # Propose code edit
    ANALYZE = "analyze"  # LLM analysis
    VERIFY = "verify"  # Syntax/semantic check
    REPORT = "report"  # Return result to user


@dataclass
class Observation:
    """Result of an action."""
    action: ActionType
    result: str
    confidence: float = 0.5
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Thought:
    """Intermediate reasoning step."""
    reasoning: str
    next_action: ActionType
    action_params: Dict[str, Any]
    confidence: float = 0.5
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ReActStep:
    """Single step in ReAct loop: observe → thought → action."""
    step_num: int
    observation: Optional[Observation]
    thought: Optional[Thought]
    action_executed: bool = False
    action_result: Optional[Any] = None
    error: Optional[str] = None


class ReActFramework:
    """
    Main ReAct orchestrator.
    
    Implements the loop:
    1. OBSERVE: Analyze query + context
    2. THOUGHT: Reason about next step
    3. ACTION: Execute tool/LLM call
    4. Repeat until goal achieved
    """
    
    def __init__(self, llm_client: LocalLLMClient = None, repo_path: str = "."):
        self.llm_client = llm_client or LocalLLMClient()
        self.streaming_executor = SyncStreamingExecutor(self.llm_client)
        self.repo_path = repo_path
        self.safe_editor = SafeEditor(repo_path)
        self.max_steps = 5  # Was 10; reduce for responsiveness
        self.no_progress_limit = 3  # number of consecutive non-progress steps
        self._no_progress_count = 0
        self._last_action = None
        self.steps: List[ReActStep] = []
        self.goal = ""
        self.trace = []  # For logging/visualization
        
    def initialize(self, user_query: str, initial_context: Dict[str, Any] = None):
        """Initialize ReAct loop with user query."""
        self.goal = user_query
        self.steps = []
        self.trace = []
        self.initial_context = initial_context or {}
        
        # Initial observation: parse query
        initial_obs = self._parse_query(user_query, self.initial_context)
        
        step = ReActStep(
            step_num=0,
            observation=initial_obs,
            thought=None
        )
        self.steps.append(step)
        self._log_trace(f"[OBSERVE] Query: {user_query}")
        
    def _parse_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Observation:
        """Parse user query to understand intent."""
        # Identify query type
        query_lower = query.lower()
        
        query_type = "unknown"
        if any(w in query_lower for w in ["find", "search", "locate"]):
            query_type = "search"
        elif any(w in query_lower for w in ["explain", "analyze", "describe"]):
            query_type = "analysis"
        elif any(w in query_lower for w in ["edit", "change", "fix", "improve"]):
            query_type = "editing"
        elif any(w in query_lower for w in ["generate", "create", "write"]):
            query_type = "generation"
        
        result = f"Query type: {query_type}. Files available: {context.get('file_count', 0)}"
        
        return Observation(
            action=ActionType.ANALYZE,
            result=result,
            confidence=0.8
        )
    
    def step(self) -> Tuple[bool, Optional[str]]:
        """
        Execute one ReAct step.
        
        Returns:
            (should_continue, error_message)
        """
        current_step_num = len(self.steps)
        
        if current_step_num >= self.max_steps:
            return False, "Max steps reached"
        
        # Get last step for context
        last_step = self.steps[-1]
        
        # THOUGHT: Reason about next action
        thought = self._generate_thought(last_step)
        
        # Detect repeated, non-progressing actions (e.g. endless search/analyze loop)
        if self._last_action == thought.next_action and thought.next_action in (ActionType.SEARCH, ActionType.ANALYZE):
            self._no_progress_count += 1
        else:
            self._no_progress_count = 0

        self._last_action = thought.next_action

        if self._no_progress_count >= self.no_progress_limit:
            self._log_trace("[GOAL STOP] No progress after several steps – stopping ReAct loop.")
            return False, "No progress – early stop"
        
        self._log_trace(f"[THOUGHT {current_step_num}] {thought.reasoning[:100]}...")
        
        # ACTION: Execute the thought
        action_result, error = self._execute_action(thought)
        
        if error:
            self._log_trace(f"[ERROR] {error}")
            return False, error
        
        # OBSERVE: Analyze result
        observation = Observation(
            action=thought.next_action,
            result=str(action_result),
            confidence=0.7
        )
        
        self._log_trace(f"[OBSERVE] {thought.next_action.value}: {action_result[:100]}...")
        
        # Create step
        step = ReActStep(
            step_num=current_step_num,
            observation=observation,
            thought=thought,
            action_executed=True,
            action_result=action_result
        )
        
        self.steps.append(step)
        
        # Check if goal achieved
        if self._is_goal_achieved(observation):
            self._log_trace(f"[GOAL ACHIEVED] After {current_step_num} steps")
            return False, None  # Don't continue
        
        return True, None  # Continue
    
    def _generate_thought(self, last_step: ReActStep) -> Thought:
        """Use LLM to reason about next action."""
        
        # Build context for LLM
        context = self._build_thought_context(last_step)
        
        thought_prompt = f"""You are a code analysis agent using the ReAct framework.

Goal: {self.goal}

Current context:
{context}

Based on the goal and context, decide the next action.

Respond in this exact JSON format:
{{
    "reasoning": "Your reasoning about what to do next",
    "next_action": "search|retrieve|edit|analyze|verify|report",
    "action_params": {{"param1": "value1", "param2": "value2"}},
    "confidence": 0.85
}}

Be concise and specific. Choose the action that best advances toward the goal."""
        
        response, success = self.streaming_executor.stream_llm_analysis_to_stdout(thought_prompt)
        if not success:
            # Fallback: non-streaming call
            response = self.llm_client.generate_text(thought_prompt)
        
        # Parse LLM response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                thought_data = json.loads(json_match.group())
            else:
                # Fallback
                thought_data = {
                    "reasoning": response,
                    "next_action": "analyze",
                    "action_params": {},
                    "confidence": 0.5
                }
        except:
            thought_data = {
                "reasoning": response,
                "next_action": "analyze",
                "action_params": {},
                "confidence": 0.5
            }
        
        # Convert action string to enum
        action_str = thought_data.get("next_action", "analyze").lower()
        try:
            action = ActionType[action_str.upper()]
        except:
            action = ActionType.ANALYZE

        # TEMPORARY: disable EDIT until we wire in code generation
        if action == ActionType.EDIT:
            action = ActionType.ANALYZE
        
        return Thought(
            reasoning=thought_data.get("reasoning", ""),
            next_action=action,
            action_params=thought_data.get("action_params", {}),
            confidence=thought_data.get("confidence", 0.5)
        )
    
    def _build_thought_context(self, last_step: ReActStep) -> str:
        """Build context for thought generation."""
        context_parts = []

        context_parts.append(f"Previous observation: {last_step.observation.result[:200]}")

        # Show action history
        if len(self.steps) > 1:
            actions_taken = [
                f"{s.thought.next_action.value if s.thought else 'none'}"
                for s in self.steps[:-1]
            ]
            context_parts.append(f"Actions taken: {', '.join(actions_taken)}")

        # Add a brief view of retrieved code chunks (if any)
        chunks = (self.initial_context or {}).get("retrieved_chunks", [])
        if chunks:
            snippet_lines = []
            for c in chunks[:3]:  # limit for brevity
                snippet_lines.append(
                    f"File: {c['file_path']}\n```{c['language']}\n{c['content'][:300]}\n```"
                )
            context_parts.append("Relevant code snippets:\n" + "\n\n".join(snippet_lines))

        return "\n".join(context_parts)
    
    def _execute_action(self, thought: Thought) -> Tuple[Any, Optional[str]]:
        """Execute the action specified in thought."""
        
        action = thought.next_action
        params = thought.action_params or {}
        
        try:
            if action == ActionType.SEARCH:
                # params: {"query": "...", "regex": false}
                query = params.get("query", self.goal)
                regex = bool(params.get("regex", False))
                results = code_search.search_code(self.repo_path, query, regex=regex)
                return results, None
            
            elif action == ActionType.RETRIEVE:
                # params: {"file_path": "relative/or/url"}
                file_path = params.get("file_path")
                if not file_path:
                    return None, "Missing file_path for RETRIEVE"
                # Local path relative to repo
                abs_path = os.path.join(self.repo_path, file_path)
                content = file_io.read_file(abs_path)
                return {"file_path": file_path, "content": content}, None
            
            elif action == ActionType.ANALYZE:
                # Call LLM for analysis
                prompt = params.get("prompt", "Analyze the context")
                result, success = self.streaming_executor.stream_llm_analysis_to_stdout(prompt)
                if not success:
                    return None, "LLM streaming failed during ANALYZE"
                return result, None
            
            elif action == ActionType.EDIT:
                # Propose an edit, do NOT apply here
                file_path = params.get("file_path")
                proposed_content = params.get("proposed_content")
                description = params.get("description", "Code change")
                reasoning = params.get("reasoning", "")
                
                if not file_path or proposed_content is None:
                    return None, "Missing file_path or proposed_content for EDIT"
                
                is_valid, proposal, error = self.safe_editor.propose_edit(
                    file_path=file_path,
                    proposed_content=proposed_content,
                    description=description,
                    reasoning=reasoning,
                    affected_lines=params.get("affected_lines", (0, 0)),
                )
                if not is_valid:
                    return None, error
                # Return proposal as a dict; CLI/Executor can later ask for confirmation
                return asdict(proposal), None
            
            elif action == ActionType.VERIFY:
                # Simple placeholder; could integrate SyntaxValidator or tests here
                return "Syntax valid", None
            
            elif action == ActionType.REPORT:
                return params.get("result", "Task complete"), None
            
            else:
                return None, f"Unknown action: {action}"
        
        except Exception as e:
            return None, str(e)
    
    def _is_goal_achieved(self, observation: Observation) -> bool:
        """Heuristic check if goal achieved."""
        # For now, simple heuristic: if we got a REPORT action, we're done
        return observation.action == ActionType.REPORT
    
    def _log_trace(self, message: str):
        """Log trace for visualization."""
        self.trace.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })
        print(message)  # Also print for visibility
    
    def print_trace(self):
        """Print full reasoning trace."""
        print("\n" + "="*70)
        print("📋 REASONING TRACE")
        print("="*70)
        for entry in self.trace:
            print(f"[{entry['timestamp']}] {entry['message']}")
        print("="*70 + "\n")
    
    def export_trace(self) -> Dict[str, Any]:
        """Export trace as structured data."""
        return {
            "goal": self.goal,
            "steps": [asdict(s) for s in self.steps],
            "trace": self.trace,
            "total_steps": len(self.steps)
        }


class ReActRunner:
    """High-level runner for complete ReAct execution."""
    
    def __init__(self, llm_client: LocalLLMClient = None, repo_path: str = "."):
        self.framework = ReActFramework(llm_client, repo_path)
    
    def execute(
        self,
        query: str,
        initial_context: Dict[str, Any] = None,
        verbose: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute complete ReAct loop until goal achieved.
        
        Returns:
            (final_result, trace_data)
        """
        self.framework.initialize(query, initial_context)

        result = None
        while True:
            should_continue, error = self.framework.step()
            if error or not should_continue:
                if self.framework.steps:
                    last_step = self.framework.steps[-1]
                    result = last_step.action_result

                    # If result is None or still not a summary, create a simple report
                    if result is None:
                        summary = f"Stopped after {len(self.framework.steps) - 1} steps. Reason: {error or 'Max steps reached'}."
                        result = summary
                break

        if verbose:
            self.framework.print_trace()

        return result, self.framework.export_trace()
