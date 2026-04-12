"""
IT Mental Health OpenEnv - Typed Pydantic models.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MentalHealthAction(BaseModel):
    response: str
    task_id: str = "burnout_detection"
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MentalHealthReward(BaseModel):
    value: float
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class MentalHealthObservation(BaseModel):
    scenario: str
    feedback: str
    task_id: str = "burnout_detection"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MentalHealthState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    current_task: str = "burnout_detection"
    cumulative_reward: float = 0.0
    tasks_completed: List[str] = Field(default_factory=list)
    task_scores: Dict[str, float] = Field(default_factory=dict)
    task_step_counts: Dict[str, int] = Field(default_factory=dict)
    difficulty_scores: Dict[str, float] = Field(default_factory=dict)
