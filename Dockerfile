# Headless, CPU-only image for training / evaluation / recording.
# Your code and outputs are NOT baked in: `make docker-run` bind-mounts the repo at /app,
# so everything the container writes (runs/, videos/, ...) lands in your local folder.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app:/app/src \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Only the dependency manifests are copied, so this layer is cached until they change.
# uv.lock is used when present (commit it after the first `make setup`).
COPY pyproject.toml uv.loc[k] ./
RUN uv sync

CMD ["bash"]
