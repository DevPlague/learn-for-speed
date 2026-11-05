# CPU-only image for training / evaluation / recording. Headless unless `make docker-play`
# passes your display in. No Rust needed on your machine, the image builds the core itself.
# Your code and outputs are NOT baked in: `make docker-run` bind-mounts the repo at /app,
# so everything the container writes (runs/, videos/, ...) lands in your local folder.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH="/opt/venv/bin:/opt/cargo/bin:$PATH" \
    PYTHONPATH=/app:/app/src \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    MPLBACKEND=Agg

# gcc links the Rust core, curl fetches the toolchain, the X libraries are for `make docker-play`.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev curl ca-certificates \
       libx11-6 libxext6 libxcursor1 libxi6 libxrandr2 libxfixes3 libxrender1 libxss1 \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --no-modify-path

WORKDIR /app

# Only the manifests and the Rust core are copied, so this layer is cached until they change.
# Changing rust/ needs `make docker-build` again, the repo mounted at /app is not rebuilt.
# uv.lock is used when present (commit it after the first `make setup`).
COPY pyproject.toml uv.loc[k] ./
COPY rust ./rust
RUN uv sync

CMD ["bash"]
