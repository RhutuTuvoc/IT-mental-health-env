"""
IT Mental Health OpenEnv - Environment Logic  (v2 - improved)

FIX 1: LLM-as-judge grader replaces keyword matching.
FIX 2: Randomised episode generation — every reset() is unique.
"""

import os
import uuid
import random
import json
from typing import Optional
from openai import OpenAI

try:
    from models import MentalHealthAction, MentalHealthObservation, MentalHealthReward, MentalHealthState
except ImportError:
    try:
        from server.models import MentalHealthAction, MentalHealthObservation, MentalHealthReward, MentalHealthState
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import MentalHealthAction, MentalHealthObservation, MentalHealthReward, MentalHealthState

TASK_ORDER = ["burnout_detection", "stress_triage", "intervention_plan"]
TASK_DIFFICULTY = {
    "burnout_detection": "easy",
    "stress_triage": "medium",
    "intervention_plan": "hard",
}

# ── LLM Judge client ──────────────────────────────────────────────────────────
_llm_client: Optional[OpenAI] = None

def _get_llm_client() -> Optional[OpenAI]:
    global _llm_client
    use_llm_judge = os.environ.get("USE_LLM_JUDGE", "").strip().lower() in {"1", "true", "yes"}
    if not use_llm_judge:
        return None
    if _llm_client is None:
        api_key = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
        if api_key:
            _llm_client = OpenAI(api_key=api_key, base_url=base_url)
    return _llm_client


# ── Randomised scenario data pools ───────────────────────────────────────────
NAMES = ["Alex","Jordan","Sam","Riley","Morgan","Casey","Taylor","Drew","Jamie","Avery"]
ROLES = ["Software Engineer","DevOps Engineer","QA Engineer","Data Scientist",
         "Backend Developer","Frontend Developer","ML Engineer","Platform Engineer"]
YEARS_EXP = [1,2,3,4,5,7,8,10]

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
    "Low":      {"hours_range":(42,50), "vacation_months":(1,4),  "dimensions":1},
    "Moderate": {"hours_range":(50,60), "vacation_months":(4,8),  "dimensions":2},
    "High":     {"hours_range":(60,68), "vacation_months":(8,14), "dimensions":2},
    "Critical": {"hours_range":(68,80), "vacation_months":(14,24),"dimensions":3},
}

STRESS_TIER_TEMPLATES = {
    "GREEN":    ["Feeling slightly overwhelmed with the sprint but I think I'll manage."],
    "AMBER":    ["Haven't slept great — around 6 hours most nights. Starting to affect focus."],
    "RED":      ["I snapped at a teammate yesterday. I'm scared of how I'm feeling lately."],
    "CRITICAL": ["I've been having chest tightness every time I get a pager alert. Three weeks. Haven't told anyone."],
}


def generate_burnout_scenario(rng):
    name = rng.choice(NAMES)
    role = rng.choice(ROLES)
    exp = rng.choice(YEARS_EXP)
    severity_label, cfg = rng.choice(list(SEVERITY_MAP.items()))
    hours = rng.randint(*cfg["hours_range"])
    vacation = rng.randint(*cfg["vacation_months"])
    n_dims = cfg["dimensions"]

    all_dims = ["exhaustion","depersonalization","reduced_accomplishment"]
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
    sym_text = "\n".join(f"- {s}" for s in symptoms)

    scenario = f"""[TASK: Burnout Detection — EASY]

You are an occupational psychologist reviewing an IT employee profile.

Employee Profile:
- Name: {name} ({role}, {exp} years experience)
- Working hours: ~{hours} hrs/week for the past 3 months
- Last vacation: {vacation} months ago
- Observed symptoms:
{sym_text}

Your task:
1. Identify which MBI dimensions are present: Exhaustion / Depersonalization / Reduced Personal Accomplishment
2. Rate severity: Low / Moderate / High / Critical
3. List TOP 3 red-flag signals from the profile
4. State whether immediate HR escalation is needed (Yes/No) and why

Respond with clear headings."""

    gt = {"active_dimensions":ground_truth_dims, "severity":severity_label,
          "escalation_needed":severity_label in ("High","Critical"),
          "name":name, "hours":hours, "vacation_months":vacation}
    return scenario, gt


def generate_stress_triage_scenario(rng):
    tiers = ["CRITICAL","RED",rng.choice(["GREEN","AMBER"])]
    rng.shuffle(tiers)
    selected_names = rng.sample(NAMES, len(tiers))
    employees = []
    for idx, tier in enumerate(tiers):
        employees.append({
            "name": selected_names[idx],
            "role": rng.choice(ROLES),
            "tier": tier,
            "quote": rng.choice(STRESS_TIER_TEMPLATES[tier]),
        })
    tier_order = {"CRITICAL":0,"RED":1,"AMBER":2,"GREEN":3}
    priority_order = sorted(range(3), key=lambda i: tier_order[employees[i]["tier"]])

    cases = "".join(
        f'{i+1}. {e["name"]} ({e["role"]}): "{e["quote"]}"\n'
        for i,e in enumerate(employees)
    )
    scenario = f"""[TASK: Stress Triage — MEDIUM]

You are reviewing urgent mental health flags from an IT pulse survey.
Triage these 3 employees and prioritise them.

Cases:
{cases}
For each case:
a) Assign: GREEN / AMBER / RED / CRITICAL
b) Primary stressor: Workload / Physiological / Relationship / Cognitive
c) ONE immediate action (within 24 hours)
d) ONE medium-term support (within 2 weeks)

Rank the 3 cases by intervention priority (1 = most urgent)."""

    gt = {
        "employees": employees,
        "correct_tiers": {e["name"]:e["tier"] for e in employees},
        "priority_order": [employees[i]["name"] for i in priority_order],
        "names": [e["name"] for e in employees],
    }
    return scenario, gt


def generate_intervention_plan_scenario(rng):
    team_size = rng.choice([8,10,12,15,18,20])
    affected = int(team_size * rng.choice([0.4,0.5,0.6,0.7]))
    overtime = rng.choice([10,15,18,20,25])
    oncall = rng.choice([2,3,4,5])
    no_tb = rng.choice([6,9,12,15,18,24])
    no_1on1 = rng.choice([3,4,5,6,8])
    on_leave = rng.randint(1,3)
    hr_c = rng.randint(1,4)

    scenario = f"""[TASK: Intervention Plan — HARD]

You are a workplace mental health consultant. Design a 4-week intervention for a
software team with systemic burnout ({affected} of {team_size} members affected).

Team Context:
- Overtime: {overtime} hrs/week above contract
- On-call: 1 person every {oncall} days (24/7)
- No team-building in {no_tb} months
- No manager 1:1s in {no_1on1} months
- {hr_c} formal HR complaints about workload
- {on_leave} members on anxiety-related medical leave

Design a 4-week plan:
Week 1: Immediate stabilisation
Week 2: Assessment & listening
Week 3: Process reforms (on-call, overtime, workload)
Week 4: Sustainable systems

For each week: 2+ concrete actions, responsible party (HR/Manager/EAP), measurable outcome.

Also include:
- 3 measurable KPIs for 90-day tracking
- 1 key risk if plan is NOT executed
- Budget: Low (<$500) / Medium ($500-$5000) / High (>$5000)"""

    gt = {"team_size":team_size,"affected":affected,"overtime":overtime,
          "oncall_days":oncall,"on_leave":on_leave,"hr_complaints":hr_c}
    return scenario, gt


# ── Clinical rubrics ──────────────────────────────────────────────────────────
RUBRICS = {
    "burnout_detection": {
        "dimensions_identified": {"max":3.0,"desc":"1pt per correct MBI dimension identified (0-3)"},
        "severity_accuracy":     {"max":2.0,"desc":"2=exact match, 1=adjacent, 0=wrong"},
        "red_flags_quality":     {"max":2.0,"desc":"Flags are specific and drawn from the profile (0-2)"},
        "escalation_reasoning":  {"max":2.0,"desc":"Correct Yes/No + sound clinical reasoning (0-2)"},
        "structure_clarity":     {"max":1.0,"desc":"Clear headings, professional format (0-1)"},
    },
    "stress_triage": {
        "tier_accuracy":       {"max":3.0,"desc":"1pt per correct tier assigned (0-3)"},
        "priority_ranking":    {"max":2.0,"desc":"2=correct, 1=partially correct, 0=wrong"},
        "immediate_actions":   {"max":2.0,"desc":"Specific 24h actions proportionate to tier (0-2)"},
        "medium_term_support": {"max":2.0,"desc":"Realistic 2-week support for each case (0-2)"},
        "completeness":        {"max":1.0,"desc":"All 3 employees addressed with all 4 elements (0-1)"},
    },
    "intervention_plan": {
        "four_week_structure":  {"max":3.0,"desc":"All 4 weeks with distinct focus (0-3)"},
        "action_concreteness":  {"max":2.0,"desc":"Actions specific, not vague (0-2)"},
        "responsibility":       {"max":1.0,"desc":"Responsible party per action (0-1)"},
        "kpis_quality":         {"max":2.0,"desc":"3 measurable, relevant KPIs (0-2)"},
        "risk_and_budget":      {"max":1.0,"desc":"Risk realistic + budget category present (0-1)"},
        "proportionality":      {"max":1.0,"desc":"Plan addresses this team's specific numbers (0-1)"},
    },
}

JUDGE_SYSTEM = """You are a clinical rubric evaluator for an AI mental health RL environment.
Score the agent's response against the rubric. Award marks based on reasoning quality and accuracy — NOT keyword presence alone.
Return ONLY valid JSON, no markdown fences:
{"scores": {"dim_name": float, ...}, "feedback": "2-3 sentence critique"}"""


def _llm_judge_grade(task_id, scenario, response, ground_truth):
    rubric = RUBRICS[task_id]
    total_max = sum(v["max"] for v in rubric.values())
    rubric_text = "\n".join(f'  "{k}": max={v["max"]} — {v["desc"]}' for k,v in rubric.items())

    prompt = f"""Task: {task_id}
Scenario: {scenario}
Ground truth: {json.dumps(ground_truth)}
Agent response: {response}
Rubric:
{rubric_text}
Score now. JSON only."""

    client = _get_llm_client()
    if client is None:
        return _heuristic_grade(task_id, response, ground_truth)

    model = os.environ.get("MODEL_NAME","meta-llama/Llama-3.1-8B-Instruct")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":JUDGE_SYSTEM},
                      {"role":"user","content":prompt}],
            max_tokens=400, temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        scores = parsed.get("scores",{})
        feedback = parsed.get("feedback","")
        breakdown, total = {}, 0.0
        for dim, cfg in rubric.items():
            s = max(0.0, min(cfg["max"], float(scores.get(dim, 0.0))))
            breakdown[dim] = round(s/cfg["max"], 3)
            total += s
        reward = round(max(0.0, min(1.0, total/total_max)), 3)
        return reward, breakdown, f"[LLM Judge] {feedback}"
    except Exception as e:
        r,b,f = _heuristic_grade(task_id, response, ground_truth)
        return r, b, f"[Fallback — judge error: {e}] {f}"


def _heuristic_grade(task_id, response, ground_truth):
    """Ground-truth-aware heuristic fallback (better than pure keyword matching)."""
    r = response.lower()
    breakdown, score = {}, 0.0

    if task_id == "burnout_detection":
        dims = ground_truth.get("active_dimensions",[])
        # Match on a distinctive token per dimension, not just the first word.
        dim_keys = {
            "Exhaustion": ["exhaustion","exhausted"],
            "Depersonalization": ["depersonalization","depersonalisation","cynicism","cynical"],
            "Reduced Personal Accomplishment": ["reduced personal accomplishment","reduced accomplishment","inefficacy","personal accomplishment"],
        }
        dim_hits = sum(1 for d in dims if any(k in r for k in dim_keys.get(d, [d.lower()])))
        breakdown["dimensions_identified"] = dim_hits/max(len(dims),1)
        # Severity: exact match = 1.0, adjacent tier = 0.5, otherwise 0.
        sev_order = ["low","moderate","high","critical"]
        sev = ground_truth.get("severity","High").lower()
        mentioned = [s for s in sev_order if s in r]
        if sev in mentioned:
            breakdown["severity_accuracy"] = 1.0
        elif mentioned and sev in sev_order:
            gt_idx = sev_order.index(sev)
            best = min(abs(sev_order.index(m) - gt_idx) for m in mentioned)
            breakdown["severity_accuracy"] = 0.5 if best == 1 else 0.0
        else:
            breakdown["severity_accuracy"] = 0.0
        breakdown["red_flags_quality"] = min(1.0, len(response)/300)
        esc = ground_truth.get("escalation_needed",True)
        esc_hit = "yes" in r or "escalat" in r or "immediate" in r
        breakdown["escalation_reasoning"] = 1.0 if esc==esc_hit else 0.3
        breakdown["structure_clarity"] = 1.0 if any(h in response for h in ["##","**","1.","1:"]) else 0.3

    elif task_id == "stress_triage":
        tiers = ground_truth.get("correct_tiers",{})
        tier_hits = sum(1 for nm,t in tiers.items() if nm.lower() in r and t.lower() in r)
        breakdown["tier_accuracy"] = tier_hits/max(len(tiers),1)
        priority = ground_truth.get("priority_order",[])
        breakdown["priority_ranking"] = 1.0 if (priority and priority[0].lower() in r[:500]) else 0.4
        breakdown["immediate_actions"] = min(1.0, r.count("24")*0.3 + r.count("immediat")*0.4)
        breakdown["medium_term_support"] = 1.0 if ("week" in r or "fortnight" in r) else 0.3
        names = ground_truth.get("names",[])
        breakdown["completeness"] = 1.0 if all(n.lower() in r for n in names) else 0.4

    elif task_id == "intervention_plan":
        weeks = sum(1 for w in ["week 1","week 2","week 3","week 4"] if w in r)
        breakdown["four_week_structure"] = weeks/4.0
        breakdown["action_concreteness"] = min(1.0, len(response)/600)
        breakdown["responsibility"] = 1.0 if any(x in r for x in ["hr","manager","eap"]) else 0.0
        kpi_hits = sum(1 for k in ["kpi","metric","measure","indicator"] if k in r)
        breakdown["kpis_quality"] = min(1.0, kpi_hits*0.4)
        breakdown["risk_and_budget"] = 1.0 if ("risk" in r and any(b in r for b in ["low","medium","high","$"])) else 0.3
        gt_size = str(ground_truth.get("team_size",""))
        breakdown["proportionality"] = 1.0 if gt_size in response else 0.5

    score = sum(breakdown.values())/max(len(breakdown),1)
    reward = round(max(0.0, min(1.0, score)), 3)
    return reward, breakdown, f"[Heuristic] {task_id} = {reward:.2f}. Breakdown: {breakdown}"


# ── Environment class ─────────────────────────────────────────────────────────
class ITMentalHealthEnvironment:
    """
    OpenEnv RL environment — IT sector mental health.
    Every reset() is unique (randomised profiles/scenarios).
    Graded by LLM-as-judge against clinical rubric.
    """
    def __init__(self):
        self._state = MentalHealthState()
        self._task_index = 0
        self._rng = random.Random()
        self._current_scenarios: dict = {}

    def _generate_all(self):
        self._current_scenarios = {
            "burnout_detection": generate_burnout_scenario(self._rng),
            "stress_triage":     generate_stress_triage_scenario(self._rng),
            "intervention_plan": generate_intervention_plan_scenario(self._rng),
        }

    def _score_metadata(self, task: str, reward: float):
        difficulty = TASK_DIFFICULTY[task]
        task_step_count = self._state.task_step_counts.get(task, 0) + 1
        task_cumulative_score = round(self._state.task_scores.get(task, 0.0) + reward, 3)
        difficulty_cumulative_score = round(
            self._state.difficulty_scores.get(difficulty, 0.0) + reward, 3
        )
        return {
            "difficulty": difficulty,
            "step_score": round(reward, 3),
            "task_step_number": task_step_count,
            "task_cumulative_score": task_cumulative_score,
            "difficulty_cumulative_score": difficulty_cumulative_score,
            "overall_cumulative_score": round(self._state.cumulative_reward + reward, 3),
        }

    def reset(self, seed=None):
        actual_seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(actual_seed)
        self._task_index = 0
        self._generate_all()
        self._state = MentalHealthState(
            episode_id=str(uuid.uuid4()), step_count=0,
            current_task=TASK_ORDER[0], cumulative_reward=0.0, tasks_completed=[],
            task_scores={task_id: 0.0 for task_id in TASK_ORDER},
            task_step_counts={task_id: 0 for task_id in TASK_ORDER},
            difficulty_scores={"easy": 0.0, "medium": 0.0, "hard": 0.0},
        )
        task = TASK_ORDER[0]
        scenario_text, _ = self._current_scenarios[task]
        return MentalHealthObservation(
            scenario=scenario_text, feedback=f"New episode (seed={actual_seed}).",
            task_id=task,
            metadata={
                "seed": actual_seed,
                "difficulty": TASK_DIFFICULTY[task],
                "step_score": 0.0,
                "task_step_number": 0,
                "task_cumulative_score": 0.0,
                "difficulty_cumulative_score": 0.0,
                "overall_cumulative_score": 0.0,
            },
        )

    def step(self, action):
        if not self._state.episode_id or not self._current_scenarios:
            raise ValueError("Episode not initialized. Call /reset before /step.")

        # Guard: refuse to grade once the episode is finished.
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
        scenario_text, ground_truth = self._current_scenarios.get(task, ("", {}))

        reward, breakdown, feedback = _llm_judge_grade(
            task_id=task, scenario=scenario_text,
            response=action.response, ground_truth=ground_truth,
        )
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
