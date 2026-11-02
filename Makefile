.DEFAULT_GOAL := help
SHELL := /bin/bash

IMAGE ?= racing-rl
CMD   ?= python --version

# ---- Local (uv) -----------------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

.PHONY: setup
setup: ## Create the Python 3.12 env with uv (first run writes uv.lock: commit it)
	@command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync

.PHONY: run
run: ## Run any command in the env, e.g. make run CMD="python train.py"
	uv run $(CMD)

.PHONY: test
test: ## Run pytest
	uv run pytest

.PHONY: lint
lint: ## Lint with ruff
	uv run ruff check .

.PHONY: format
format: ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: tb
tb: ## TensorBoard on http://localhost:6006 (reads ./runs)
	uv run tensorboard --logdir runs --port 6006

.PHONY: clean
clean: ## Remove caches (keeps runs/, videos/, models/)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ---- Docker (headless; the repo folder is bind-mounted, so outputs are saved locally) --------
# Runs as your own user so files written to ./runs etc. are not owned by root.
DOCKER_RUN = docker run --rm -it --user $$(id -u):$$(id -g) \
	-v "$(CURDIR)":/app -w /app -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl $(IMAGE)

.PHONY: docker-build
docker-build: ## Build the image
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

.PHONY: docker-tb
docker-tb: ## TensorBoard from Docker on http://localhost:6006
	docker run --rm -p 6006:6006 --user $$(id -u):$$(id -g) -v "$(CURDIR)":/app -w /app \
		-e HOME=/tmp $(IMAGE) tensorboard --logdir runs --host 0.0.0.0 --port 6006
