"""
Static task registry for OpenEnv task discoverability.
"""

TASKS = [
    {
        "id": "burnout_detection",
        "difficulty": "easy",
        "objective": "Identify burnout dimensions, severity, top red flags, and whether immediate HR escalation is needed.",
        "grader": "burnout_detection_grader",
        "graders": ["burnout_detection_grader"],
    },
    {
        "id": "stress_triage",
        "difficulty": "medium",
        "objective": "Classify three employee cases by stress tier, rank urgency, and recommend immediate plus 2-week support.",
        "grader": "stress_triage_grader",
        "graders": ["stress_triage_grader"],
    },
    {
        "id": "intervention_plan",
        "difficulty": "hard",
        "objective": "Create a four-week intervention plan with owners, KPIs, risk, and budget for a burned-out team.",
        "grader": "intervention_plan_grader",
        "graders": ["intervention_plan_grader"],
    },
]

TASK_IDS = [task["id"] for task in TASKS]

__all__ = ["TASKS", "TASK_IDS"]
