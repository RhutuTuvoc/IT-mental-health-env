"""
IT Mental Health OpenEnv - Environment Logic.

This environment defines three tasks with deterministic programmatic graders:
- burnout_detection (easy)
- stress_triage (medium)
- intervention_plan (hard)
"""

import random
import uuid

from graders import grade_response

try:
    from models import MentalHealthObservation, MentalHealthReward, MentalHealthState
except ImportError:
    try:
        from server.models import MentalHealthObservation, MentalHealthReward, MentalHealthState
    except ImportError:
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import MentalHealthObservation, MentalHealthReward, MentalHealthState


TASK_ORDER = ["burnout_detection", "stress_triage", "intervention_plan"]
TASK_DIFFICULTY = {
    "burnout_detection": "easy",
    "stress_triage": "medium",
    "intervention_plan": "hard",
}


NAMES = ["Alex", "Jordan", "Sam", "Riley", "Morgan", "Casey", "Taylor", "Drew", "Jamie", "Avery"]
ROLES = [
    "Software Engineer",
    "DevOps Engineer",
    "QA Engineer",
    "Data Scientist",
    "Backend Developer",
    "Frontend Developer",
    "ML Engineer",
    "Platform Engineer",
]
YEARS_EXP = [1, 2, 3, 4, 5, 7, 8, 10]

EXHAUSTION_SYMPTOMS = [
    "sleeping only 4-5 hours/night",
    "skipping meals regularly",
    "frequent headaches and migraines",
    "working 65-75 hours/week for 3+ months",
    "no vacation in over 12 months",
    "falling asleep at desk",
    "constant tiredness even after weekends",
]
DEPERSONALIZATION_SYMPTOMS = [
    "says 'I don't care about outcomes anymore'",
    "becoming cynical about project goals",
    "stopped attending team standups",
    "avoids messages for hours",
    "emotionally detached during sprint reviews",
    "sarcastic about the company mission",
]
REDUCED_ACCOMPLISHMENT_SYMPTOMS = [
    "submitting incomplete pull requests",
    "missing sprint commitments regularly",
    "says 'nothing I do matters anyway'",
    "code review quality has declined sharply",
    "stopped proposing solutions in meetings",
    "productivity metrics down 40% vs last quarter",
]

SEVERITY_MAP = {
    "Low": {"hours_range": (42, 50), "vacation_months": (1, 4), "dimensions": 1},
    "Moderate": {"hours_range": (50, 60), "vacation_months": (4, 8), "dimensions": 2},
    "High": {"hours_range": (60, 68), "vacation_months": (8, 14), "dimensions": 2},
    "Critical": {"hours_range": (68, 80), "vacation_months": (14, 24), "dimensions": 3},
}

STRESS_TIER_TEMPLATES = {
    "GREEN": ["Feeling slightly overwhelmed with the sprint but I think I'll manage."],
    "AMBER": ["Haven't slept great - around 6 hours most nights. Starting to affect focus."],
    "RED": ["I snapped at a teammate yesterday. I'm scared of how I'm feeling lately."],
    "CRITICAL": ["I've been having chest tightness every time I get a pager alert. Three weeks. Haven't told anyone."],
}


def generate_burnout_scenario(rng: random.Random):
    name = rng.choice(NAMES)
    role = rng.choice(ROLES)
    exp = rng.choice(YEARS_EXP)
    severity_label, cfg = rng.choice(list(SEVERITY_MAP.items()))
    hours = rng.randint(*cfg["hours_range"])
    vacation = rng.randint(*cfg["vacation_months"])
    n_dims = cfg["dimensions"]

    all_dims = ["exhaustion", "depersonalization", "reduced_accomplishment"]
    active_dims = rng.sample(all_dims, n_dims)
    symptoms = []
    ground_truth_dims = []
    if "exhaustion" in active_dims:
        symptoms += rng.sample(EXHAUSTION_SYMPTOMS, 2)
        ground_truth_dims.append("Exhaustion")
    if "depersonalization" in active_dims:
        symptoms += rng.sample(DEPERSONALIZATION_SYMPTOMS, 2)
        ground_truth_dims.append("Depersonalization")
    if "reduced_accomplishment" in active_dims:
        symptoms += rng.sample(REDUCED_ACCOMPLISHMENT_SYMPTOMS, 2)
        ground_truth_dims.append("Reduced Personal Accomplishment")
    rng.shuffle(symptoms)
    sym_text = "\n".join(f"- {item}" for item in symptoms)

    scenario = f"""[TASK: Burnout Detection - EASY]

You are an occupational psychologist reviewing an IT employee profile.

Employee Profile:
- Name: {name} ({role}, {exp} years experience)
- Working hours: ~{hours} hrs/week for the past 3 months
- Last vacation: {vacation} months ago
- Observed symptoms:
{sym_text}

Objective:
1. Identify which MBI dimensions are present
2. Rate severity: Low / Moderate / High / Critical
3. List the top 3 red-flag signals
4. State whether immediate HR escalation is needed and why

Respond with clear headings."""

    ground_truth = {
        "active_dimensions": ground_truth_dims,
        "severity": severity_label,
        "escalation_needed": severity_label in ("High", "Critical"),
        "name": name,
        "hours": hours,
        "vacation_months": vacation,
    }
    return scenario, ground_truth


def generate_stress_triage_scenario(rng: random.Random):
    tiers = ["CRITICAL", "RED", rng.choice(["GREEN", "AMBER"])]
    rng.shuffle(tiers)
    selected_names = rng.sample(NAMES, len(tiers))
    employees = []
    for idx, tier in enumerate(tiers):
        employees.append(
            {
                "name": selected_names[idx],
                "role": rng.choice(ROLES),
                "tier": tier,
                "quote": rng.choice(STRESS_TIER_TEMPLATES[tier]),
            }
        )

    tier_order = {"CRITICAL": 0, "RED": 1, "AMBER": 2, "GREEN": 3}
    priority_order = sorted(range(3), key=lambda index: tier_order[employees[index]["tier"]])
    cases = "".join(
        f'{index + 1}. {employee["name"]} ({employee["role"]}): "{employee["quote"]}"\n'
        for index, employee in enumerate(employees)
    )
    scenario = f"""[TASK: Stress Triage - MEDIUM]

You are reviewing urgent mental health flags from an IT pulse survey.
Triage these 3 employees and prioritise them.

Cases:
{cases}
Objective:
a) Assign: GREEN / AMBER / RED / CRITICAL
b) Name the primary stressor
c) Give one immediate action within 24 hours
d) Give one medium-term support action within 2 weeks

Rank the 3 cases by intervention priority (1 = most urgent)."""

    ground_truth = {
        "employees": employees,
        "correct_tiers": {employee["name"]: employee["tier"] for employee in employees},
        "priority_order": [employees[index]["name"] for index in priority_order],
        "names": [employee["name"] for employee in employees],
    }
    return scenario, ground_truth


def generate_intervention_plan_scenario(rng: random.Random):
    team_size = rng.choice([8, 10, 12, 15, 18, 20])
    affected = int(team_size * rng.choice([0.4, 0.5, 0.6, 0.7]))
    overtime = rng.choice([10, 15, 18, 20, 25])
    oncall = rng.choice([2, 3, 4, 5])
    no_tb = rng.choice([6, 9, 12, 15, 18, 24])
    no_1on1 = rng.choice([3, 4, 5, 6, 8])
    on_leave = rng.randint(1, 3)
    hr_complaints = rng.randint(1, 4)

    scenario = f"""[TASK: Intervention Plan - HARD]

You are a workplace mental health consultant. Design a 4-week intervention for a
software team with systemic burnout ({affected} of {team_size} members affected).

Team Context:
- Overtime: {overtime} hrs/week above contract
- On-call: 1 person every {oncall} days (24/7)
- No team-building in {no_tb} months
- No manager 1:1s in {no_1on1} months
- {hr_complaints} formal HR complaints about workload
- {on_leave} members on anxiety-related medical leave

Objective:
- Week 1: Immediate stabilisation
- Week 2: Assessment and listening
- Week 3: Process reforms
- Week 4: Sustainable systems

For each week include 2+ actions, responsible owner, measurable outcome.
Also include 3 KPIs, 1 key risk, and a budget category."""

    ground_truth = {
        "team_size": team_size,
        "affected": affected,
        "overtime": overtime,
        "oncall_days": oncall,
        "on_leave": on_leave,
        "hr_complaints": hr_complaints,
    }
    return scenario, ground_truth


class ITMentalHealthEnvironment:
    """OpenEnv-compatible environment with deterministic task graders."""

    def __init__(self):
        self._state = MentalHealthState()
        self._task_index = 0
        self._rng = random.Random()
        self._current_scenarios: dict = {}

    def _generate_all(self):
        self._current_scenarios = {
            "burnout_detection": generate_burnout_scenario(self._rng),
            "stress_triage": generate_stress_triage_scenario(self._rng),
            "intervention_plan": generate_intervention_plan_scenario(self._rng),
        }

    def _score_metadata(self, task: str, reward: float):
        difficulty = TASK_DIFFICULTY[task]
        task_step_count = self._state.task_step_counts.get(task, 0) + 1
        task_cumulative_score = round(self._state.task_scores.get(task, 0.0) + reward, 3)
        difficulty_cumulative_score = round(self._state.difficulty_scores.get(difficulty, 0.0) + reward, 3)
        return {
            "difficulty": difficulty,
            "step_score": round(reward, 3),
            "task_step_number": task_step_count,
            "task_cumulative_score": task_cumulative_score,
            "difficulty_cumulative_score": difficulty_cumulative_score,
            "overall_cumulative_score": round(self._state.cumulative_reward + reward, 3),
            "task_id": task,
        }

    def reset(self, seed=None):
        actual_seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(actual_seed)
        self._task_index = 0
        self._generate_all()
        self._state = MentalHealthState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task=TASK_ORDER[0],
            cumulative_reward=0.0,
            tasks_completed=[],
            task_scores={task_id: 0.0 for task_id in TASK_ORDER},
            task_step_counts={task_id: 0 for task_id in TASK_ORDER},
            difficulty_scores={"easy": 0.0, "medium": 0.0, "hard": 0.0},
        )
        task = TASK_ORDER[0]
        scenario_text, _ = self._current_scenarios[task]
        return MentalHealthObservation(
            scenario=scenario_text,
            feedback=f"New episode (seed={actual_seed}).",
            task_id=task,
            metadata={
                "seed": actual_seed,
                "difficulty": TASK_DIFFICULTY[task],
                "step_score": 0.0,
                "task_step_number": 0,
                "task_cumulative_score": 0.0,
                "difficulty_cumulative_score": 0.0,
                "overall_cumulative_score": 0.0,
                "task_id": task,
            },
        )

    def step(self, action):
        if not self._state.episode_id or not self._current_scenarios:
            raise ValueError("Episode not initialized. Call /reset before /step.")

        if self._task_index >= len(TASK_ORDER):
            observation = MentalHealthObservation(
                scenario="Episode already finished. Call /reset to start a new one.",
                feedback="No-op: episode is done.",
                task_id=self._state.current_task,
            )
            reward = MentalHealthReward(value=0.0, score_breakdown={}, feedback="No-op: episode is done.")
            return observation, reward, True, {}

        self._state.step_count += 1
        task = self._state.current_task
        if action.task_id != task:
            raise ValueError(f"task_id mismatch: expected '{task}', got '{action.task_id}'.")

        _, ground_truth = self._current_scenarios.get(task, ("", {}))
        reward, breakdown, feedback = grade_response(task, action.response, ground_truth)

        score_metadata = self._score_metadata(task, reward)
        self._state.cumulative_reward += reward
        self._state.task_scores[task] = score_metadata["task_cumulative_score"]
        self._state.task_step_counts[task] = score_metadata["task_step_number"]
        self._state.difficulty_scores[TASK_DIFFICULTY[task]] = score_metadata["difficulty_cumulative_score"]
        self._state.tasks_completed.append(task)

        self._task_index += 1
        if self._task_index >= len(TASK_ORDER):
            done = True
            next_task = task
            next_scenario = "All tasks complete. Episode finished."
        else:
            done = False
            next_task = TASK_ORDER[self._task_index]
            self._state.current_task = next_task
            next_scenario, _ = self._current_scenarios[next_task]

        observation = MentalHealthObservation(
            scenario=next_scenario,
            feedback=feedback,
            task_id=next_task,
            metadata=score_metadata,
        )
        reward_model = MentalHealthReward(
            value=reward,
            score_breakdown=breakdown,
            feedback=feedback,
        )
        return observation, reward_model, done, score_metadata

    def state(self):
        return self._state
