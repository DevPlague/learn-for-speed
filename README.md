<p align="center">
  <img src="docs/media/demo.gif" alt="Pure pursuit driving a validation track, lidar rays and minimap visible" width="560">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Rust-stable-B7410E?logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/Gymnasium-env-0081A5" alt="Gymnasium">
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-a97bff" alt="Stable-Baselines3 PPO">
  <br>
  <img src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

<h1 align="center">learn-for-speed</h1>

<p align="center">A 2D top-down racing car that <b>learns to drive circuits it has never seen</b>, using reinforcement learning.</p>

## 🏁 Introduction

learn-for-speed generates random closed circuits and trains a PPO agent to lap them as fast as it can.

- The car sees the world only through its own sensors. A simulated lidar with 25 rays, plus its speed, steering and sideways slip. No map, no centerline, no GPS.
- Physics are a bicycle model written in Rust. The car grips in slow corners, drifts in fast ones and can reverse out of trouble.
- Every result is measured on **held-out tracks** the agent never trained on, against a random driver and a hand-written pure pursuit controller.

The GIF above is the pure pursuit baseline on validation track 1000. The yellow lines are the lidar rays the agent gets as input.

## Contents

- [Features](#-features)
- [Install](#-install)
- [Usage](#-usage)
- [Make commands](#-make-commands)
- [Driving keys](#driving-keys)
- [Seeds and splits](#seeds-and-splits)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#-contributing)
- [To Do](#to-do)

## ✨ Features

**Simulation**

- Procedural closed spline tracks with a difficulty knob from 0 (gentle) to 1 (tight corners).
- Rust physics core, one call per step with NumPy arrays, so training stays fast and headless.
- Grass is allowed but slow and slippery. Barriers end the attempt.
- Drive any track yourself with the keyboard.

**Learning**

- Custom Gymnasium env with a 28 value observation in [-1, 1] and a 2 value action, steer and throttle or brake.
- Reward for progress along the centerline, with penalties for time, steering jitter, grass, standing still, crashing and getting stuck, and a bonus for each lap.
- PPO from Stable-Baselines3 on CPU, parallel envs, one YAML config per experiment, checkpoints and a validation curve during training.

**Evaluation**

- Random and pure pursuit baselines.
- Lap success, off-track and stuck rates, lap time, speed and progress, as a Markdown table and JSON, with mean ± std over training seeds.

**Dashboard**

- One static page built from `runs/`. Learning curves, validation tables, synced replays of every policy with lidar and follow camera, and physics plots. No server needed.

## 📦 Install

Pick one. Local is faster and needs Rust. Docker needs nothing but Docker.

**Option A, local on Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                            # uv, the Python tool
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal -c clippy -c rustfmt     # Rust, for the physics core
git clone git@github.com:DevPlague/learn-for-speed.git
cd learn-for-speed
make setup     # first run downloads dependencies and compiles Rust, a few minutes
make test      # everything should pass
```

**Option B, Docker**

Install Docker from [the official guide](https://docs.docker.com/engine/install/). Your user must be able to run `docker` without `sudo`.

```bash
git clone git@github.com:DevPlague/learn-for-speed.git
cd learn-for-speed
make docker-build   # again whenever rust/, pyproject.toml or uv.lock change
make docker-test
```

Your folder is mounted into the container, so Python changes apply at once and everything written to `runs/` lands on your disk.

## 🚦 Usage

The whole project in six steps. Each one is a single command, run from the repo folder.

| Step | Local | Docker |
|---|---|---|
| 1. Drive a track yourself | `make play` | `make docker-play` |
| 2. Train an agent | `make train CONFIG=tiny` | `make docker-run CMD="python -m learning.train --config configs/tiny.yaml"` |
| 3. Watch it learn | `make tb` | `make docker-tb` |
| 4. Evaluate it | `make eval POLICY=ppo MODEL=runs/tiny_s0/best_model.zip` | `make docker-run CMD="python -m evaluation.evaluate --policy ppo --model runs/tiny_s0/best_model.zip --split val --n 20"` |
| 5. Compare with the baselines | `make eval POLICY=pure_pursuit` and `make eval POLICY=random` | same with `--policy pure_pursuit` or `random` |
| 6. See it all in the dashboard | `make dashboard RUNS="tiny_s0"` | `make docker-run CMD="python -m dashboard.build --runs tiny_s0"` |

**1. Drive.** A window opens on the start line. Pick any track and difficulty with `make play SEED=1002 DIFFICULTY=1`.

**2. Train.** `CONFIG` is a file name in `configs/` and `SEED` the training seed, default 0. Outputs go to `runs/<config>_s<seed>/`, with the config, `model.zip`, `best_model.zip`, checkpoints, `progress.csv` and TensorBoard logs.

| Config | What it is | Time on a laptop |
|---|---|---|
| `tiny` | 20k steps, checks that everything runs | about 1 minute |
| `probe`, `probe_nojitter` | 200k steps, quick reward experiments | a few minutes |
| `mvp` | the main run | long, ask the team first |

For mean ± std, train the same config with several seeds, for example `make train CONFIG=mvp SEED=1`.

**3. Watch.** TensorBoard opens on http://localhost:6006 and reads every run in `runs/`. Refresh it while training runs.

**4 and 5. Evaluate.** One episode per validation track, default the first 20 (`N=20`) at difficulty 0.5. Pass several models at once to average over seeds, `MODEL="runs/mvp_s0/best_model.zip runs/mvp_s1/best_model.zip"`. Results print as a table and save to `runs/eval_<policy>_val/eval.json`. The full process is in [docs/EVALUATION.md](docs/EVALUATION.md).

**6. Dashboard.** `RUNS` lists the run folders to evaluate and replay next to the baselines. Then open `runs/dashboard/index.html` in your browser.

## 🛠 Make commands

`make help` prints them all. Variables go after the command, like `make play SEED=1002`.

**Set up**

| Command | What it does |
|---|---|
| `make setup` | Creates the Python env and builds the Rust core. First time and after pulling new dependencies |
| `make rust` | Forces a Rust rebuild, only if a Rust change does not seem to apply |
| `make clean` | Removes Python caches, keeps `runs/`, `videos/` and `models/` |

**Drive, train and evaluate**

| Command | Variables | What it does |
|---|---|---|
| `make play` | `SEED` `DIFFICULTY` | Drive by keyboard |
| `make train` | `CONFIG` `SEED` | Train PPO from `configs/CONFIG.yaml` |
| `make eval` | `POLICY` `MODEL` `N` `DIFFICULTY` `SPLIT` | Evaluate `random`, `pure_pursuit` or `ppo` |
| `make run CMD="..."` | `CMD` | Run any command inside the env |

**Look at results**

| Command | Variables | What it does |
|---|---|---|
| `make tb` | | TensorBoard on http://localhost:6006 |
| `make dashboard` | `RUNS` | Build the dashboard into `runs/dashboard/` |

**Check code**

| Command | What it does |
|---|---|
| `make test` | Runs all tests |
| `make lint` | Checks Python with ruff and Rust with rustfmt and clippy |
| `make format` | Auto-formats Python and Rust |
| `make bench` | Prints env steps per second and track generation time |

**Docker**

| Command | What it does |
|---|---|
| `make docker-build` | Builds the image |
| `make docker-test` | Runs all tests in the container |
| `make docker-play` | Drives by keyboard with the window on your screen, takes `SEED` and `DIFFICULTY` |
| `make docker-run CMD="..."` | Runs any command in the container, for headless training or evaluation |
| `make docker-shell` | Opens a shell in the container |
| `make docker-tb` | TensorBoard from the container on http://localhost:6006 |

## Driving keys

| Key | Action |
|---|---|
| Up | Accelerate, or brake while reversing |
| Down | Brake, and once stopped, reverse |
| Left, Right | Steer |
| Esc | Quit |

An attempt fails after 10 s in a row on grass, on reaching the barrier, after 10 s without progress, or after 120 s. A failed attempt or a finished lap puts you back on the start line. Corners taken too fast make the car drift, as [the physics guide](docs/PHYSICS.md) explains.

## Seeds and splits

| Split | Track seeds | Used for |
|---|---|---|
| Train | 100000 and up | Training only, random every episode |
| Validation | 1000 to 1999 | Everything while developing, `make eval` default |
| Test | 10000 and up | Final results only, with `SPLIT=test`. Never tune on it |

The evaluation script rejects training seeds, so no training track reaches the results.
