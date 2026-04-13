"""
Deterministic agent graders for the IT Mental Health OpenEnv tasks.

Each task has:
- a concrete objective
- a deterministic programmatic grader
- a normalized score in the range [0.0, 1.0]

These graders are intentionally rule-based so submission validators can
statically discover them and evaluation remains reproducible.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


TaskState = Dict[str, Any]
GradeResult = Tuple[float, Dict[str, float], str]


def _normalize_reward(reward: float) -> float:
    return round(min(max(float(reward), 0.0), 1.0), 3)


def _safe_lower(value: Any) -> str:
    return str(value or "").lower()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _count_contains(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def _metadata_task_id(state: TaskState) -> str:
    metadata = state.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("task_id", ""))
    return ""


def _task_matches(state: TaskState, expected_task_id: str) -> bool:
    candidates = {
        str(state.get("task_id", "")),
        str(state.get("current_task", "")),
        _metadata_task_id(state),
    }
    return expected_task_id in candidates


def _breakdown_to_reward(
    breakdown: Dict[str, float], weights: Dict[str, float]
) -> float:
    total_weight = sum(weights.values()) or 1.0
    weighted = sum(breakdown.get(key, 0.0) * weight for key, weight in weights.items())
    return _normalize_reward(weighted / total_weight)


def _burnout_grade(response: str, ground_truth: Dict[str, Any]) -> GradeResult:
    text = _safe_lower(response)
    dims = ground_truth.get("active_dimensions", [])
    dim_tokens = {
        "Exhaustion": ["exhaustion", "exhausted", "fatigue", "tiredness"],
        "Depersonalization": ["depersonalization", "depersonalisation", "cynicism", "cynical", "detached"],
        "Reduced Personal Accomplishment": [
            "reduced personal accomplishment",
            "reduced accomplishment",
            "inefficacy",
            "ineffective",
            "personal accomplishment",
        ],
    }

    dim_hits = 0
    for dimension in dims:
        if _contains_any(text, dim_tokens.get(dimension, [dimension.lower()])):
            dim_hits += 1
    dimensions_identified = dim_hits / max(len(dims), 1)

    severity_order = ["low", "moderate", "high", "critical"]
    expected_severity = _safe_lower(ground_truth.get("severity"))
    mentioned = [level for level in severity_order if level in text]
    if expected_severity in mentioned:
        severity_accuracy = 1.0
    elif mentioned and expected_severity in severity_order:
        gt_idx = severity_order.index(expected_severity)
        min_distance = min(abs(severity_order.index(level) - gt_idx) for level in mentioned)
        severity_accuracy = 0.5 if min_distance == 1 else 0.0
    else:
        severity_accuracy = 0.0

    red_flag_terms = [
        str(ground_truth.get("hours", "")),
        str(ground_truth.get("vacation_months", "")),
        "week",
        "hours",
        "vacation",
        "leave",
        "headache",
        "migraine",
        "sleep",
        "desk",
        "detached",
        "cynical",
        "incomplete",
        "productivity",
    ]
    red_flags_quality = min(1.0, _count_contains(text, red_flag_terms) / 3.0)

    escalation_expected = bool(ground_truth.get("escalation_needed"))
    escalation_positive = _contains_any(text, ["yes", "escalat", "immediate hr", "urgent hr"])
    escalation_reasoning = 1.0 if escalation_positive == escalation_expected else 0.0

    structure_clarity = 1.0 if _contains_any(response, ["1.", "2.", "3.", "4.", "##", "**"]) else 0.4

    breakdown = {
        "dimensions_identified": round(dimensions_identified, 3),
        "severity_accuracy": round(severity_accuracy, 3),
        "red_flags_quality": round(red_flags_quality, 3),
        "escalation_reasoning": round(escalation_reasoning, 3),
        "structure_clarity": round(structure_clarity, 3),
    }
    weights = {
        "dimensions_identified": 0.30,
        "severity_accuracy": 0.20,
        "red_flags_quality": 0.20,
        "escalation_reasoning": 0.20,
        "structure_clarity": 0.10,
    }
    reward = _breakdown_to_reward(breakdown, weights)
    feedback = (
        f"[Deterministic Grader] burnout_detection score={reward:.2f}. "
        f"Expected severity={ground_truth.get('severity')}; "
        f"dimension hits={dim_hits}/{max(len(dims), 1)}."
    )
    return reward, breakdown, feedback


def _stress_triage_grade(response: str, ground_truth: Dict[str, Any]) -> GradeResult:
    text = _safe_lower(response)
    tiers = ground_truth.get("correct_tiers", {})
    names = ground_truth.get("names", [])
    priority_order = [str(name).lower() for name in ground_truth.get("priority_order", [])]

    tier_hits = 0
    for name, tier in tiers.items():
        name_lower = name.lower()
        tier_lower = tier.lower()
        if name_lower in text and tier_lower in text:
            tier_hits += 1
    tier_accuracy = tier_hits / max(len(tiers), 1)

    top_two_correct = sum(1 for name in priority_order[:2] if name and name in text[:800])
    priority_ranking = 1.0 if top_two_correct == 2 else 0.5 if top_two_correct == 1 else 0.0

    immediate_actions = 1.0 if _contains_any(text, ["24 hour", "24-hour", "within 24", "immediate", "today"]) else 0.3
    medium_term_support = 1.0 if _contains_any(text, ["2 week", "2-week", "within 2 weeks", "fortnight", "two weeks"]) else 0.3
    completeness = 1.0 if all(name.lower() in text for name in names) else 0.0

    breakdown = {
        "tier_accuracy": round(tier_accuracy, 3),
        "priority_ranking": round(priority_ranking, 3),
        "immediate_actions": round(immediate_actions, 3),
        "medium_term_support": round(medium_term_support, 3),
        "completeness": round(completeness, 3),
    }
    weights = {
        "tier_accuracy": 0.35,
        "priority_ranking": 0.20,
        "immediate_actions": 0.15,
        "medium_term_support": 0.15,
        "completeness": 0.15,
    }
    reward = _breakdown_to_reward(breakdown, weights)
    feedback = (
        f"[Deterministic Grader] stress_triage score={reward:.2f}. "
        f"Correct tier assignments={tier_hits}/{max(len(tiers), 1)}."
    )
    return reward, breakdown, feedback


def _intervention_plan_grade(response: str, ground_truth: Dict[str, Any]) -> GradeResult:
    text = _safe_lower(response)

    week_hits = sum(1 for week in ["week 1", "week 2", "week 3", "week 4"] if week in text)
    four_week_structure = week_hits / 4.0

    action_terms = [
        "on-call",
        "overtime",
        "survey",
        "1:1",
        "1-1",
        "check-in",
        "training",
        "leave",
        "workload",
        "rotation",
        "policy",
    ]
    action_concreteness = min(1.0, _count_contains(text, action_terms) / 6.0)
    responsibility = 1.0 if _contains_any(text, ["hr", "manager", "eap"]) else 0.0

    kpi_terms = ["kpi", "metric", "measure", "indicator", "tracking", "90-day"]
    kpis_quality = 1.0 if _count_contains(text, kpi_terms) >= 2 else 0.5 if _count_contains(text, kpi_terms) == 1 else 0.0

    risk_and_budget = 1.0 if ("risk" in text and _contains_any(text, ["budget", "low", "medium", "high", "$"])) else 0.0

    proportionality_terms = [
        str(ground_truth.get("team_size", "")),
        str(ground_truth.get("affected", "")),
        str(ground_truth.get("overtime", "")),
        str(ground_truth.get("oncall_days", "")),
        str(ground_truth.get("hr_complaints", "")),
    ]
    proportionality = 1.0 if _count_contains(text, [term for term in proportionality_terms if term]) >= 2 else 0.5

    breakdown = {
        "four_week_structure": round(four_week_structure, 3),
        "action_concreteness": round(action_concreteness, 3),
        "responsibility": round(responsibility, 3),
        "kpis_quality": round(kpis_quality, 3),
        "risk_and_budget": round(risk_and_budget, 3),
        "proportionality": round(proportionality, 3),
    }
    weights = {
        "four_week_structure": 0.30,
        "action_concreteness": 0.20,
        "responsibility": 0.10,
        "kpis_quality": 0.15,
        "risk_and_budget": 0.10,
        "proportionality": 0.15,
    }
    reward = _breakdown_to_reward(breakdown, weights)
    feedback = (
        f"[Deterministic Grader] intervention_plan score={reward:.2f}. "
        f"Week coverage={week_hits}/4."
    )
    return reward, breakdown, feedback


def grade_response(task_id: str, response: str, ground_truth: Dict[str, Any]) -> GradeResult:
    if task_id == "burnout_detection":
        return _burnout_grade(response, ground_truth)
    if task_id == "stress_triage":
        return _stress_triage_grade(response, ground_truth)
    if task_id == "intervention_plan":
        return _intervention_plan_grade(response, ground_truth)
    return 0.0, {}, f"[Deterministic Grader] unknown task_id={task_id}"


def grade_burnout_detection(state: TaskState, reward: float) -> float:
    if not _task_matches(state, "burnout_detection"):
        return 0.0
    return _normalize_reward(reward)


def grade_stress_triage(state: TaskState, reward: float) -> float:
    if not _task_matches(state, "stress_triage"):
        return 0.0
    return _normalize_reward(reward)


def grade_intervention_plan(state: TaskState, reward: float) -> float:
    if not _task_matches(state, "intervention_plan"):
        return 0.0
    return _normalize_reward(reward)


GRADERS = {
    "burnout_detection_grader": grade_burnout_detection,
    "stress_triage_grader": grade_stress_triage,
    "intervention_plan_grader": grade_intervention_plan,
}


TASK_GRADER_PAIRS: List[Tuple[str, str]] = [
    ("burnout_detection", "burnout_detection_grader"),
    ("stress_triage", "stress_triage_grader"),
    ("intervention_plan", "intervention_plan_grader"),
]


TASK_GRADER_OBJECTIVES = {
    "burnout_detection": "Identify burnout dimensions, severity, red flags, and escalation need from an employee profile.",
    "stress_triage": "Triage three employees by urgency and recommend immediate plus medium-term support.",
    "intervention_plan": "Produce a four-week intervention plan with owners, KPIs, risk, and budget.",
}


__all__ = [
    "GRADERS",
    "TASK_GRADER_OBJECTIVES",
    "TASK_GRADER_PAIRS",
    "grade_burnout_detection",
    "grade_intervention_plan",
    "grade_response",
    "grade_stress_triage",
]
