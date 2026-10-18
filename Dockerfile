# syntax=docker/dockerfile:1
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    BIENCODER_MODEL="jinaai/jina-embeddings-v2-base-code" \
    CROSSENCODER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2" \
    CODE_DIR="/app/src/semantic_search"

WORKDIR /app

# 1. Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 2. Copy project source
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 3. Create user and chown directories
RUN useradd --create-home -r -u 10001 appuser && chown -R appuser /app
RUN mkdir -p /opt/models && chown -R appuser /opt/models
USER appuser

# 4. Warmup: download models into the image
RUN python -m semantic_search.warmup

EXPOSE 8000
CMD ["python", "-m", "semantic_search"]
