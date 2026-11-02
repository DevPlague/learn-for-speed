# racing-rl

A 2D top-down racing car that **learns to drive circuits it has never seen**, using reinforcement learning (PPO).

Tracks are generated procedurally (random closed splines). The car senses the road with simulated lidar plus a short lookahead of the centerline, and outputs steering and throttle/brake. We train with a curriculum (easy tracks first) and evaluate on **held-out tracks** against baselines (random and a hand-written pure-pursuit controller) and ablations.

**Stack:** Python 3.12 only, `gymnasium` + `pygame` (simulator), `stable-baselines3` PPO (CPU PyTorch), `imageio` for GIF/MP4 demos.

## Quickstart (Linux)

```bash
make setup                          # uv env (first run creates uv.lock: commit it)
make run CMD="python your_script.py"
make test | make lint | make tb     # pytest, ruff, TensorBoard
```

No local install? Use Docker (headless; outputs still land in your local folder):

```bash
make docker-build
make docker-run CMD="python your_script.py"
make docker-shell
```

`make help` lists everything. The code layout is up to the team; planning lives in [`docs/PLAN.md`](docs/PLAN.md).
