"""
Inference script for IT Mental Health OpenEnv.

Required environment variables in the inference configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Optional:
    LOCAL_IMAGE_NAME  Present for compatibility with environments created from
                      docker images.

Defaults are set only for API_BASE_URL and MODEL_NAME.
The script uses the OpenAI client for all LLM calls.
"""

import os
import sys
from typing import List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "it_mental_health_env"
MAX_STEPS = 3
TEMPERATURE = 0.3
MAX_TOKENS = 800
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are an expert occupational psychologist and workplace mental health consultant
specialising in the IT/software engineering sector. You have deep knowledge of:
- Maslach Burnout Inventory (MBI) dimensions
- Stress triage frameworks (GREEN/AMBER/RED/CRITICAL)
- Evidence-based workplace mental health interventions
- Employee Assistance Programmes (EAP) and HR escalation protocols

Respond in clear, structured format with headings and concrete recommendations.
Be specific, actionable, and professional. Aim for 200-600 words per response."""


def clean_field(value: Optional[str]) -> str:
    if value is None:
        return "null"
    return " ".join(str(value).split())


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={clean_field(task)} env={clean_field(env)} model={clean_field(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={clean_field(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={clean_field(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def call_env(endpoint: str, payload: Optional[dict] = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    if method == "GET":
        response = requests.get(url, timeout=30)
    else:
        response = requests.post(url, json=payload or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_model_response(client: OpenAI, scenario: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    content = completion.choices[0].message.content or ""
    return content.strip() or "Provide a structured assessment and immediate next steps."


def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    current_task = "unknown"
    start_logged = False

    try:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN must be set for inference.")

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        observation = call_env("reset", {})
        current_task = observation.get("task_id", "unknown")
        log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)
        start_logged = True

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            current_task = observation.get("task_id", "unknown")
            action_text = get_model_response(client, observation["scenario"])

            error = None
            done = False
            reward = 0.0

            try:
                observation = call_env(
                    "step",
                    {
                        "response": action_text,
                        "task_id": current_task,
                        "confidence": 0.85,
                        "metadata": {},
                    },
                )
                reward = float(observation.get("reward", 0.0) or 0.0)
                done = bool(observation.get("done", False))
            except Exception as exc:
                error = str(exc)
                done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_text, reward=reward, done=done, error=error)

            if error or done:
                break

        score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        if not start_logged:
            log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)
