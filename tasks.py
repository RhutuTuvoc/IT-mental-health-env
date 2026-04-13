"""
Static task registry for OpenEnv task discoverability.
"""

TASKS = [
    {
        "id": "burnout_detection",
        "grader": "burnout_detection_grader",
        "graders": ["burnout_detection_grader"],
    },
    {
        "id": "stress_triage",
        "grader": "stress_triage_grader",
        "graders": ["stress_triage_grader"],
    },
    {
        "id": "intervention_plan",
        "grader": "intervention_plan_grader",
        "graders": ["intervention_plan_grader"],
    },
]


TASK_IDS = [task["id"] for task in TASKS]


__all__ = ["TASKS", "TASK_IDS"]
