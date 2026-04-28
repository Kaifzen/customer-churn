FROM python:3.11-slim

WORKDIR /app

# LightGBM runtime dependency (OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary in the container.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency manifests first to maximize layer cache reuse.
COPY pyproject.toml uv.lock* README.md ./

# Install project dependencies into the uv-managed environment.
RUN uv sync --locked --no-dev

# Copy source code and serving model artifacts.
COPY src /app/src

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Run the consolidated FastAPI app entrypoint.
CMD ["uv", "run", "uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]