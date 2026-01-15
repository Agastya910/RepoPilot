"""
Interaction Logger for collecting training data.

Stores all user interactions in a standardized format for:
1. Offline reinforcement learning (Q-learning, IQL, CQL)
2. Supervised fine-tuning
3. Analytics and improvement
4. Safety auditing
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid


@dataclass
class InteractionRecord:
    """Single interaction record for training data."""
    
    # Metadata
    interaction_id: str
    timestamp: str
    user_id: str = "anonymous"
    
    # Input
    query: str
    repo_path: str
    repo_name: str
    
    # Agent decisions
    query_type: str  # "search", "analysis", "editing", "generation"
    high_level_plan: str  # What the high-level agent decided
    file_selected: str  # Which file to work on
    
    # Code generation
    code_change_proposed: str  # The patch proposed
    code_change_description: str  # What it does
    code_change_reasoning: str  # Why it does it
    
    # Verification
    syntax_valid: bool
    imports_correct: bool
    logic_sound: bool  # User feedback
    
    # Outcome
    user_approved: bool  # Did user accept?
    user_feedback: Optional[str] = None  # User comments
    
    # Results
    success: bool  # Did it work?
    error_message: Optional[str] = None
    
    # Metrics
    tokens_used: int = 0
    latency_ms: float = 0.0
    
    # For RL training
    reward: float = 0.0  # Computed from outcome


@dataclass
class TrajectoryRecord:
    """Full conversation trajectory (multiple interactions)."""
    
    trajectory_id: str
    timestamp: str
    user_id: str = "anonymous"
    
    # Full interaction sequence
    interactions: List[InteractionRecord]
    
    # Overall success
    goal_achieved: bool
    total_reward: float = 0.0
    
    # Session info
    repo_name: str = ""
    session_length: int = 0  # Minutes


class InteractionLogger:
    """
    Logs interactions for training data collection and analytics.
    
    Directory structure:
    .repopilot/
    ├── interactions/
    │   ├── 2024-01-15/
    │   │   ├── uuid_1.json
    │   │   ├── uuid_2.json
    │   │   └── ...
    │   └── trajectories/
    │       ├── traj_uuid_1.json
    │       └── ...
    └── analytics.json
    """
    
    def __init__(self, repo_path: str, data_dir: str = ".repopilot/interactions"):
        self.repo_path = repo_path
        self.data_dir = os.path.join(repo_path, data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "trajectories"), exist_ok=True)
        
        # Create today's date directory
        today = datetime.now().strftime("%Y-%m-%d")
        self.today_dir = os.path.join(self.data_dir, today)
        os.makedirs(self.today_dir, exist_ok=True)
    
    def log_interaction(
        self,
        query: str,
        repo_name: str,
        query_type: str,
        file_selected: str,
        code_change_proposed: str,
        code_change_description: str,
        code_change_reasoning: str,
        syntax_valid: bool,
        user_approved: bool,
        success: bool,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        error_message: Optional[str] = None,
        user_feedback: Optional[str] = None,
    ) -> str:
        """
        Log a single interaction.
        
        Returns:
            interaction_id
        """
        
        # Compute reward based on outcome
        reward = self._compute_reward(
            user_approved=user_approved,
            success=success,
            syntax_valid=syntax_valid,
            error=error_message is not None
        )
        
        record = InteractionRecord(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            query=query,
            repo_path=self.repo_path,
            repo_name=repo_name,
            query_type=query_type,
            high_level_plan=query_type,  # Simplified for now
            file_selected=file_selected,
            code_change_proposed=code_change_proposed,
            code_change_description=code_change_description,
            code_change_reasoning=code_change_reasoning,
            syntax_valid=syntax_valid,
            imports_correct=True,  # TODO: Check imports
            logic_sound=user_approved,  # Proxy for logic soundness
            user_approved=user_approved,
            user_feedback=user_feedback,
            success=success,
            error_message=error_message,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            reward=reward,
        )
        
        # Save to file
        file_path = os.path.join(
            self.today_dir,
            f"{record.interaction_id}.json"
        )
        
        with open(file_path, "w") as f:
            json.dump(asdict(record), f, indent=2)
        
        return record.interaction_id
    
    def log_trajectory(
        self,
        interactions: List[str],  # interaction IDs
        goal_achieved: bool,
        repo_name: str,
        session_length_minutes: int = 0
    ) -> str:
        """
        Log a full trajectory (conversation).
        
        Returns:
            trajectory_id
        """
        
        # Load interaction records
        records = []
        total_reward = 0.0
        
        for interaction_id in interactions:
            record = self._load_interaction(interaction_id)
            if record:
                records.append(record)
                total_reward += record.reward
        
        trajectory = TrajectoryRecord(
            trajectory_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            interactions=records,
            goal_achieved=goal_achieved,
            total_reward=total_reward,
            repo_name=repo_name,
            session_length=session_length_minutes
        )
        
        # Save trajectory
        traj_path = os.path.join(
            self.data_dir,
            "trajectories",
            f"{trajectory.trajectory_id}.json"
        )
        
        with open(traj_path, "w") as f:
            # Custom serialization because of nested records
            json.dump({
                "trajectory_id": trajectory.trajectory_id,
                "timestamp": trajectory.timestamp,
                "interactions": [asdict(r) for r in records],
                "goal_achieved": trajectory.goal_achieved,
                "total_reward": trajectory.total_reward,
                "repo_name": trajectory.repo_name,
                "session_length": trajectory.session_length,
            }, f, indent=2)
        
        return trajectory.trajectory_id
    
    def _compute_reward(
        self,
        user_approved: bool,
        success: bool,
        syntax_valid: bool,
        error: bool
    ) -> float:
        """
        Compute reward signal for RL training.
        
        Reward function (can be tuned):
        - +1.0 if user approved AND success
        - +0.5 if user approved but not success
        - -0.5 if syntax error
        - -1.0 if error/crash
        """
        
        base = 0.0
        
        if error:
            return -1.0
        
        if not syntax_valid:
            base -= 0.5
        
        if user_approved:
            base += 0.5
        
        if success:
            base += 0.5
        
        return base
    
    def _load_interaction(self, interaction_id: str) -> Optional[InteractionRecord]:
        """Load interaction from storage."""
        # Search in all date directories
        for date_dir in os.listdir(self.data_dir):
            dir_path = os.path.join(self.data_dir, date_dir)
            if not os.path.isdir(dir_path) or date_dir == "trajectories":
                continue
            
            file_path = os.path.join(dir_path, f"{interaction_id}.json")
            if os.path.exists(file_path):
                with open(file_path) as f:
                    data = json.load(f)
                return InteractionRecord(**data)
        
        return None
    
    def export_training_dataset(
        self,
        output_file: str = "training_dataset.jsonl",
        min_interactions: int = 0
    ) -> int:
        """
        Export all interactions as JSONL for training.
        
        Format: One interaction per line
        
        Returns:
            Number of interactions exported
        """
        
        count = 0
        with open(output_file, "w") as out:
            for date_dir in os.listdir(self.data_dir):
                dir_path = os.path.join(self.data_dir, date_dir)
                if not os.path.isdir(dir_path) or date_dir == "trajectories":
                    continue
                
                for filename in os.listdir(dir_path):
                    if filename.endswith(".json"):
                        with open(os.path.join(dir_path, filename)) as f:
                            record = json.load(f)
                            out.write(json.dumps(record) + "\n")
                            count += 1
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics on collected data."""
        
        stats = {
            "total_interactions": 0,
            "total_trajectories": 0,
            "successful_edits": 0,
            "user_approval_rate": 0.0,
            "avg_reward": 0.0,
            "syntax_error_rate": 0.0,
        }
        
        # Count interactions
        interaction_count = 0
        success_count = 0
        approved_count = 0
        syntax_error_count = 0
        total_reward = 0.0
        
        for date_dir in os.listdir(self.data_dir):
            dir_path = os.path.join(self.data_dir, date_dir)
            if not os.path.isdir(dir_path) or date_dir == "trajectories":
                continue
            
            for filename in os.listdir(dir_path):
                if filename.endswith(".json"):
                    with open(os.path.join(dir_path, filename)) as f:
                        record = json.load(f)
                        interaction_count += 1
                        
                        if record.get("success"):
                            success_count += 1
                        if record.get("user_approved"):
                            approved_count += 1
                        if not record.get("syntax_valid"):
                            syntax_error_count += 1
                        
                        total_reward += record.get("reward", 0.0)
        
        # Count trajectories
        traj_dir = os.path.join(self.data_dir, "trajectories")
        traj_count = len(os.listdir(traj_dir)) if os.path.exists(traj_dir) else 0
        
        stats["total_interactions"] = interaction_count
        stats["total_trajectories"] = traj_count
        stats["successful_edits"] = success_count
        stats["user_approval_rate"] = (
            approved_count / interaction_count if interaction_count > 0 else 0.0
        )
        stats["avg_reward"] = (
            total_reward / interaction_count if interaction_count > 0 else 0.0
        )
        stats["syntax_error_rate"] = (
            syntax_error_count / interaction_count if interaction_count > 0 else 0.0
        )
        
        return stats
