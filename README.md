---
title: IT Mental Health OpenEnv
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Benchmark for burnout & stress triage
tags:
  - openenv
  - reinforcement-learning
  - benchmark
  - mental-health
  - llm-evaluation
  - hackathon
---

# IT Mental Health OpenEnv

An OpenEnv-compatible benchmark for workplace mental health reasoning in IT and software engineering teams. The environment evaluates whether an agent can detect burnout signals, triage urgent stress cases, and propose a realistic intervention plan for a struggling team.

## What It Evaluates

Each episode runs through three ordered tasks:

| Task ID | Difficulty | Goal |
|---|---|---|
| `burnout_detection` | Easy | Identify Maslach burnout dimensions, severity, red flags, and escalation need |
| `stress_triage` | Medium | Classify three employees by urgency and recommend immediate support |
| `intervention_plan` | Hard | Produce a four-week team intervention plan with owners, KPIs, and budget |

Every step returns:

- a normalized reward in `[0, 1]`
- rubric feedback
- a per-dimension score breakdown
- the next scenario until the episode is complete

## API

The FastAPI server exposes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /tasks`
- `GET /schema`

### Session Handling

The API now supports per-client sessions.

- Browser and `TestClient` users automatically get an `itmh_session_id` cookie after `POST /reset`.
- API clients can also pass `X-Session-Id` or include `session_id` in the `POST /reset` body.
- `POST /step` accepts the session through cookie, `X-Session-Id`, or `metadata.session_id`.
- `GET /state` accepts the session through cookie, `X-Session-Id`, or `?session_id=...`.

For backward compatibility, stateless validators still work through a fallback anonymous session.

### Example `POST /reset`

```json
{
  "seed": 123
}
```

### Example `POST /step`

```json
{
  "response": "1. Burnout dimensions\nExhaustion and depersonalization are present.\n\n2. Severity\nHigh.\n\n3. Red flags\nLong working hours, prolonged lack of leave, emotional detachment.\n\n4. HR escalation\nYes. The combination of sustained overload and disengagement warrants prompt support.",
  "task_id": "burnout_detection",
  "confidence": 0.9,
  "metadata": {}
}
```

## Project Layout

```text
.
|-- app.py
|-- server/
|   |-- __init__.py
|   `-- app.py
|-- it_mental_health_environment.py
|-- models.py
|-- inference.py
|-- validate.py
|-- openenv.yaml
|-- Dockerfile
`-- tests/
```

`app.py` is a compatibility shim for tools that still import `app:app`. The real FastAPI implementation lives in `server/app.py`.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_inference.txt
```

Start the API:

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```

Run the validator:

```bash
python validate.py
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Inference Configuration

`inference.py` reads:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_BASE_URL`
- `LOCAL_IMAGE_NAME` (optional compatibility field)

The script uses the OpenAI client, logs one `[START]` line, one `[STEP]` line per action, and one `[END]` line even on failure.

## Docker and Hugging Face Spaces

Build locally:

```bash
docker build -t it-mental-health-env .
```

Run locally:

```bash
docker run -p 7860:7860 it-mental-health-env
```

The container now starts `uvicorn server.app:app`, honors the `PORT` environment variable, and includes health checks for Spaces-style deployment.

## Notes

- This benchmark is for structured evaluation, not clinical diagnosis.
- If no judge model is configured, grading falls back to a heuristic scorer.
- The benchmark is designed for reproducible evaluation, not personalized counseling.

## License

MIT
