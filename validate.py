"""
validate.py - Pre-submission validator for IT Mental Health OpenEnv
Run this before submitting: python validate.py

Checks:
1. openenv.yaml is present and has required fields
2. /health returns 200
3. /reset responds correctly
4. /step responds correctly (all 3 tasks)
5. /state responds correctly
6. Rewards are in 0.0-1.0 range
7. inference.py exists at repo root
"""

import os
import sys

import requests

BASE_URL = "http://localhost:7860"

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    results.append(ok)
    return ok


def run():
    print("\n" + "=" * 60)
    print("  IT Mental Health OpenEnv - Pre-Submission Validator")
    print("=" * 60 + "\n")

    try:
        import yaml

        with open("openenv.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        required = ["spec_version", "name", "type", "runtime", "app", "port", "tasks"]
        missing = [k for k in required if k not in cfg]
        check(
            "openenv.yaml present & valid",
            not missing,
            f"missing: {missing}" if missing else f"name={cfg['name']}",
        )
        task_count = len(cfg.get("tasks", []))
        check("3+ tasks in openenv.yaml", task_count >= 3, f"{task_count} tasks found")
    except Exception as e:
        check("openenv.yaml", False, str(e))

    check("inference.py at root", os.path.exists("inference.py"))

    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        check("/health returns 200", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        check("/health", False, str(e))

    try:
        r = requests.post(f"{BASE_URL}/reset", json={}, timeout=10)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        check("/reset returns 200", ok)
        check("/reset has 'scenario'", "scenario" in data)
        check("/reset has 'reward'", "reward" in data)
        check("/reset has 'done'", "done" in data)
    except Exception as e:
        check("/reset", False, str(e))

    try:
        requests.post(f"{BASE_URL}/reset", json={}, timeout=10)
        task_ids = ["burnout_detection", "stress_triage", "intervention_plan"]
        sample_response = (
            "The employee shows Exhaustion, Depersonalization, and Reduced Personal Accomplishment. "
            "Severity: High/Critical. Red flags: 70hr weeks, 14 months no leave, disconnection. "
            "Immediate HR escalation: Yes - employee is at critical burnout risk.\n\n"
            "Jordan: RED/CRITICAL tier. Riley: RED tier (chest tightness = physiological alarm). "
            "Sam: AMBER tier. Priority ranking: 1=Riley, 2=Jordan, 3=Sam.\n\n"
            "Week 1: Immediate on-call freeze, HR 1:1s for all affected. Responsible: HR Manager. "
            "Week 2: Anonymous team survey and individual 30min check-ins. Responsible: HR. "
            "Week 3: On-call rotation reform, overtime cap policy. Responsible: Manager. "
            "Week 4: Bi-weekly 1:1s, EAP enrollment, manager training. Responsible: HR + External EAP. "
            "KPIs: overtime hours, eNPS score, sick leave frequency. "
            "Risk: further attrition and possible disability claims. Budget: Medium ($500-$5000)."
        )
        for tid in task_ids:
            r = requests.post(
                f"{BASE_URL}/step",
                json={
                    "response": sample_response,
                    "task_id": tid,
                    "confidence": 0.9,
                },
                timeout=30,
            )
            ok = r.status_code == 200
            check(f"/step task={tid} returns 200", ok)
            if ok:
                d = r.json()
                reward = d.get("reward", -1)
                valid = 0.0 <= reward <= 1.0
                check(f"reward in [0,1] for {tid}", valid, f"reward={reward:.3f}")
    except Exception as e:
        check("/step", False, str(e))

    try:
        r = requests.get(f"{BASE_URL}/state", timeout=10)
        ok = r.status_code == 200
        check("/state returns 200", ok)
        if ok:
            d = r.json()
            check("/state has episode_id", "episode_id" in d)
            check("/state has step_count", "step_count" in d)
    except Exception as e:
        check("/state", False, str(e))

    total = len(results)
    passed = sum(results)
    print(f"\n{'=' * 60}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  ALL CHECKS PASSED - Ready to submit!")
    else:
        print("  Fix failing checks before submitting.")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run()
