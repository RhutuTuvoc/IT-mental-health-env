"""
IT Mental Health OpenEnv - FastAPI server.

Exposes /reset, /step, /state, /health, /tasks, and /schema endpoints.
"""

from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from it_mental_health_environment import ITMentalHealthEnvironment, TASK_ORDER
from models import MentalHealthAction

app = FastAPI(
    title="IT Mental Health OpenEnv",
    description="RL environment for mental health assessment in the IT sector.",
    version="1.0.0",
)

SESSION_COOKIE_NAME = "itmh_session_id"
ANONYMOUS_SESSION_ID = "anonymous"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


class SessionStore:
    """Thread-safe registry of environment instances keyed by session id."""

    def __init__(self):
        self._lock = Lock()
        self._sessions: Dict[str, ITMentalHealthEnvironment] = {
            ANONYMOUS_SESSION_ID: ITMentalHealthEnvironment()
        }

    def get_or_create(self, session_id: str) -> ITMentalHealthEnvironment:
        with self._lock:
            env = self._sessions.get(session_id)
            if env is None:
                env = ITMentalHealthEnvironment()
                self._sessions[session_id] = env
            return env

    def reset_session(self, session_id: str, seed: Optional[int] = None):
        env = self.get_or_create(session_id)
        return env.reset(seed=seed)

    def clear(self) -> None:
        with self._lock:
            self._sessions = {ANONYMOUS_SESSION_ID: ITMentalHealthEnvironment()}


session_store = SessionStore()

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
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    response: str
    task_id: Optional[str] = "burnout_detection"
    confidence: Optional[float] = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObservationResponse(BaseModel):
    scenario: str
    feedback: str
    reward: float
    done: bool
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    task_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    episode_id: Optional[str]
    step_count: int
    current_task: str
    cumulative_reward: float
    tasks_completed: list
    session_id: str


def _preferred_session_id(
    request: Request,
    explicit_session_id: Optional[str] = None,
    header_session_id: Optional[str] = None,
) -> Optional[str]:
    return explicit_session_id or header_session_id or request.cookies.get(SESSION_COOKIE_NAME)


def _response_metadata(metadata: Optional[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
    payload = dict(metadata or {})
    payload["session_id"] = session_id
    return payload


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
    return {
        "status": "ok",
        "env": "it_mental_health_env",
        "version": "1.0.0",
        "session_cookie": SESSION_COOKIE_NAME,
    }


@app.post("/reset", response_model=ObservationResponse)
def reset(
    response: Response,
    request: Request,
    req: ResetRequest = ResetRequest(),
    x_session_id: Optional[str] = Header(default=None),
):
    requested_session_id = _preferred_session_id(
        request=request,
        explicit_session_id=req.session_id,
        header_session_id=x_session_id,
    )
    session_id = requested_session_id or str(uuid4())

    obs = session_store.reset_session(session_id, seed=req.seed)
    actual_seed = obs.metadata.get("seed")

    # Preserve compatibility for stateless validators that do not reuse cookies.
    if requested_session_id is None:
        session_store.reset_session(ANONYMOUS_SESSION_ID, seed=actual_seed)

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
    )

    return ObservationResponse(
        scenario=obs.scenario,
        feedback=obs.feedback,
        reward=obs.reward,
        done=obs.done,
        score_breakdown=obs.score_breakdown,
        task_id=obs.task_id,
        metadata=_response_metadata(obs.metadata, session_id),
    )


@app.post("/step", response_model=ObservationResponse)
def step(
    req: StepRequest,
    request: Request,
    x_session_id: Optional[str] = Header(default=None),
):
    session_id = _preferred_session_id(
        request=request,
        explicit_session_id=req.metadata.get("session_id"),
        header_session_id=x_session_id,
    ) or ANONYMOUS_SESSION_ID
    env = session_store.get_or_create(session_id)
    action = MentalHealthAction(
        response=req.response,
        task_id=req.task_id or "burnout_detection",
        confidence=req.confidence or 1.0,
        metadata=req.metadata or {},
    )
    try:
        obs = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ObservationResponse(
        scenario=obs.scenario,
        feedback=obs.feedback,
        reward=obs.reward,
        done=obs.done,
        score_breakdown=obs.score_breakdown,
        task_id=obs.task_id,
        metadata=_response_metadata(obs.metadata, session_id),
    )


@app.get("/state", response_model=StateResponse)
def state(
    request: Request,
    session_id: Optional[str] = Query(default=None),
    x_session_id: Optional[str] = Header(default=None),
):
    resolved_session_id = _preferred_session_id(
        request=request,
        explicit_session_id=session_id,
        header_session_id=x_session_id,
    ) or ANONYMOUS_SESSION_ID
    env = session_store.get_or_create(resolved_session_id)
    s = env.state
    return StateResponse(
        episode_id=s.episode_id,
        step_count=s.step_count,
        current_task=s.current_task,
        cumulative_reward=s.cumulative_reward,
        tasks_completed=s.tasks_completed,
        session_id=resolved_session_id,
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


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
