"""
IT Mental Health OpenEnv - FastAPI Server
Exposes /reset, /step, /state, /health, /tasks endpoints.
"""

from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from it_mental_health_environment import ITMentalHealthEnvironment, TASK_ORDER
from models import MentalHealthAction

app = FastAPI(
    title="IT Mental Health OpenEnv",
    description="RL environment for mental health assessment in the IT sector.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single session for Spaces)
env = ITMentalHealthEnvironment()

TASK_METADATA = {
    "burnout_detection": {
        "difficulty": "easy",
        "description": "Identify Maslach burnout dimensions, severity, red flags, and whether HR escalation is needed.",
    },
    "stress_triage": {
        "difficulty": "medium",
        "description": "Triage three IT employees by stress tier and recommend immediate and medium-term support.",
    },
    "intervention_plan": {
        "difficulty": "hard",
        "description": "Design a four-week intervention plan for a software team facing systemic burnout.",
    },
}


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    response: str
    task_id: Optional[str] = "burnout_detection"
    confidence: Optional[float] = 1.0
    metadata: Optional[Dict[str, Any]] = {}


class ObservationResponse(BaseModel):
    scenario: str
    feedback: str
    reward: float
    done: bool
    score_breakdown: Dict[str, float] = {}
    task_id: str
    metadata: Dict[str, Any] = {}


class StateResponse(BaseModel):
    episode_id: Optional[str]
    step_count: int
    current_task: str
    cumulative_reward: float
    tasks_completed: list


@app.get("/")
def root():
    return {
        "name": "IT Mental Health OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "it_mental_health_env", "version": "1.0.0"}


@app.post("/reset", response_model=ObservationResponse)
def reset(req: ResetRequest = ResetRequest()):
    obs = env.reset(seed=req.seed)
    return ObservationResponse(
        scenario=obs.scenario,
        feedback=obs.feedback,
        reward=obs.reward,
        done=obs.done,
        score_breakdown=obs.score_breakdown,
        task_id=obs.task_id,
        metadata=obs.metadata,
    )


@app.post("/step", response_model=ObservationResponse)
def step(req: StepRequest):
    action = MentalHealthAction(
        response=req.response,
        task_id=req.task_id or "burnout_detection",
        confidence=req.confidence or 1.0,
        metadata=req.metadata or {},
    )
    obs = env.step(action)
    return ObservationResponse(
        scenario=obs.scenario,
        feedback=obs.feedback,
        reward=obs.reward,
        done=obs.done,
        score_breakdown=obs.score_breakdown,
        task_id=obs.task_id,
        metadata=obs.metadata,
    )


@app.get("/state", response_model=StateResponse)
def state():
    s = env.state
    return StateResponse(
        episode_id=s.episode_id,
        step_count=s.step_count,
        current_task=s.current_task,
        cumulative_reward=s.cumulative_reward,
        tasks_completed=s.tasks_completed,
    )


@app.get("/tasks")
def list_tasks():
    """List all tasks with metadata used by grader validation."""
    return {
        "tasks": [
            {
                "task_id": tid,
                "difficulty": TASK_METADATA[tid]["difficulty"],
                "description": TASK_METADATA[tid]["description"],
            }
            for tid in TASK_ORDER
        ]
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "response": "str - agent's textual answer",
            "task_id": "str - one of: burnout_detection, stress_triage, intervention_plan",
            "confidence": "float 0.0-1.0",
            "metadata": "dict - optional",
        },
        "observation": {
            "scenario": "str - current scenario text",
            "feedback": "str - grader feedback",
            "reward": "float 0.0-1.0",
            "done": "bool",
            "score_breakdown": "dict - partial scores per rubric dimension",
            "task_id": "str",
        },
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
