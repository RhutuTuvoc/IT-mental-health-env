FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY server/ ./server/
COPY it_mental_health_environment.py .
COPY models.py .
COPY inference.py .
COPY validate.py .
COPY openenv.yaml .
COPY README.md .
COPY app.py .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers"]
