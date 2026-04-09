FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY app.py /app/app.py
COPY server /app/server
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

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers"]
