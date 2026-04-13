"""
Static grader registry for OpenEnv task discoverability.

The environment computes the real rubric score in `it_mental_health_environment.py`.
These helpers expose one named grader per task so submission validators can
statically discover at least three task-grader pairs.
"""

from typing import Any, Dict


def _normalize_reward(reward: float) -> float:
    return min(max(float(reward), 0.0), 1.0)


def _task_matches(state: Dict[str, Any], expected_task_id: str) -> bool:
    if not isinstance(state, dict):
        return False

    candidates = [
        state.get("task_id"),
        state.get("current_task"),
        (state.get("metadata") or {}).get("task_id") if isinstance(state.get("metadata"), dict) else None,
    ]
    return expected_task_id in candidates


def grade_burnout_detection(state: Dict[str, Any], reward: float) -> float:
    return _normalize_reward(reward if _task_matches(state, "burnout_detection") else 0.0)


def grade_stress_triage(state: Dict[str, Any], reward: float) -> float:
    return _normalize_reward(reward if _task_matches(state, "stress_triage") else 0.0)


def grade_intervention_plan(state: Dict[str, Any], reward: float) -> float:
    return _normalize_reward(reward if _task_matches(state, "intervention_plan") else 0.0)


GRADERS = {
    "burnout_detection_grader": grade_burnout_detection,
    "stress_triage_grader": grade_stress_triage,
    "intervention_plan_grader": grade_intervention_plan,
}


TASK_GRADER_PAIRS = [
    ("burnout_detection", "burnout_detection_grader"),
    ("stress_triage", "stress_triage_grader"),
    ("intervention_plan", "intervention_plan_grader"),
]


__all__ = [
    "grade_burnout_detection",
    "grade_stress_triage",
    "grade_intervention_plan",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]
