# IT Mental Health OpenEnv Dockerfile
# Compatible with Hugging Face Spaces (port 7860)

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app.py /app/app.py
COPY it_mental_health_environment.py /app/it_mental_health_environment.py
COPY models.py /app/models.py
COPY inference.py /app/inference.py
COPY validate.py /app/validate.py
COPY openenv.yaml /app/openenv.yaml
COPY README.md /app/README.md
COPY .env.example /app/.env.example

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
