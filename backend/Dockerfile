# CUDA runtime for SDXL. Python 3.12 + venv provided by uv (jammy apt has no 3.12),
# installed OUTSIDE /app so the compose bind-mount of ./backend can't shadow it.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Resolve deps + fetch a managed CPython 3.12 (both land outside /app).
# Source is bind-mounted at runtime, so don't install the project itself.
COPY pyproject.toml uv.lock* ./
RUN uv sync --python 3.12 --no-install-project

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000
# Overridden by compose (adds --reload); safe default for standalone runs.
CMD ["uvicorn", "app.infrastructure:app", "--host", "0.0.0.0", "--port", "8000"]
