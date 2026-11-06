.DEFAULT_GOAL := help
SHELL := /bin/bash

IMAGE ?= racing-rl
CMD   ?= python --version
SEED  ?= 1000
DIFFICULTY ?= 0.5
RUNS  ?=

# rustup installs here, and a new login is needed before the shell knows about it
export PATH := $(HOME)/.cargo/bin:$(PATH)
RUST_HELP = Rust not found. Install: curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal -c clippy -c rustfmt

# ---- Local (uv) -----------------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

.PHONY: setup
setup: ## Create the Python env and build the Rust core (needs uv and Rust)
	@command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	@command -v cargo >/dev/null || { echo "$(RUST_HELP)"; exit 1; }
	uv sync

.PHONY: rust
rust: ## Force a rebuild of the Rust core (normally automatic)
	uv sync --reinstall-package racing-core

.PHONY: play
play: ## Drive the car by keyboard, e.g. make play SEED=1002 DIFFICULTY=1
	uv run python -m sim.play --seed $(SEED) --difficulty $(DIFFICULTY)

.PHONY: run
run: ## Run any command in the env, e.g. make run CMD="python train.py"
	uv run $(CMD)

.PHONY: test
test: ## Run all tests with pytest
	uv run pytest

.PHONY: lint
lint: ## Check Python (ruff) and Rust (rustfmt, clippy)
	uv run ruff check .
	uv run ruff format --check .
	cargo fmt --manifest-path rust/Cargo.toml --check
	cargo clippy --manifest-path rust/Cargo.toml -- -D warnings

.PHONY: format
format: ## Auto-format Python and Rust
	uv run ruff format .
	uv run ruff check --fix .
	cargo fmt --manifest-path rust/Cargo.toml

.PHONY: bench
bench: ## Print env steps per second and track generation time
	uv run python -m sim.bench

.PHONY: tb
tb: ## TensorBoard on http://localhost:6006 (reads ./runs)
	uv run tensorboard --logdir runs --port 6006

.PHONY: dashboard
dashboard: ## Build the local dashboard into runs/dashboard, e.g. make dashboard RUNS="mvp_s0"
	uv run python -m dashboard.build --runs $(RUNS)

.PHONY: clean
clean: ## Remove caches (keeps runs/, videos/, models/)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ---- Docker (headless; the repo folder is bind-mounted, so outputs are saved locally) --------
# Runs as your own user so files written to ./runs etc. are not owned by root.
DOCKER_RUN = docker run --rm -it --user $$(id -u):$$(id -g) \
	-v "$(CURDIR)":/app -w /app -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl $(IMAGE)

.PHONY: docker-build
docker-build: ## Build the image, again after any change in rust/
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run: ## Run a command in Docker, e.g. make docker-run CMD="python train.py"
	$(DOCKER_RUN) $(CMD)

.PHONY: docker-shell
docker-shell: ## Interactive shell in the container
	$(DOCKER_RUN) bash

.PHONY: docker-test
docker-test: ## Run pytest in Docker
	$(DOCKER_RUN) pytest

.PHONY: docker-play
docker-play: ## Drive by keyboard from Docker, window on your screen, e.g. make docker-play SEED=1002
	@xhost +SI:localuser:$$(id -un) >/dev/null
	docker run --rm -it --user $$(id -u):$$(id -g) -v "$(CURDIR)":/app -w /app -e HOME=/tmp \
		-e DISPLAY="$$DISPLAY" -e SDL_VIDEODRIVER=x11 -v /tmp/.X11-unix:/tmp/.X11-unix \
		$(IMAGE) python -m sim.play --seed $(SEED) --difficulty $(DIFFICULTY)

.PHONY: docker-tb
docker-tb: ## TensorBoard from Docker on http://localhost:6006
	docker run --rm -p 6006:6006 --user $$(id -u):$$(id -g) -v "$(CURDIR)":/app -w /app \
		-e HOME=/tmp $(IMAGE) tensorboard --logdir runs --host 0.0.0.0 --port 6006
