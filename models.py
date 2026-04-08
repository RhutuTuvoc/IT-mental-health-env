"""
IT Mental Health OpenEnv - Typed Models
Action / Observation / State for the IT Burnout & Mental Wellness environment.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class MentalHealthAction:
    """
    Action submitted by the agent for each mental health scenario step.

    Fields:
        response (str): The agent's textual response / intervention recommendation.
        task_id (str): Which task is being attempted: 'burnout_detection',
                       'stress_triage', or 'intervention_plan'.
        confidence (float): Agent's self-reported confidence 0.0–1.0.
        metadata (dict): Optional extra fields (reasoning chain, flags, etc.)
    """
    response: str
    task_id: str = "burnout_detection"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MentalHealthObservation:
    """
    Observation returned after each step.

    Fields:
        scenario (str): The current scenario description shown to the agent.
        feedback (str): Evaluator feedback on the last action.
        reward (float): Reward for the last step (0.0 – 1.0).
        done (bool): Whether the episode is finished.
        score_breakdown (dict): Partial scores by rubric dimension.
        task_id (str): Current task identifier.
    """
    scenario: str
    feedback: str
    reward: float
    done: bool
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    task_id: str = "burnout_detection"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MentalHealthState:
    """
    Episode state / metadata.

    Fields:
        episode_id (str): UUID for the current episode.
        step_count (int): How many steps have elapsed.
        current_task (str): Active task identifier.
        cumulative_reward (float): Total reward accumulated so far.
        tasks_completed (list): List of task IDs already completed.
    """
    episode_id: Optional[str] = None
    step_count: int = 0
    current_task: str = "burnout_detection"
    cumulative_reward: float = 0.0
    tasks_completed: List[str] = field(default_factory=list)
