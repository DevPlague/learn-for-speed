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
- [Commands](#-commands)
- [Tracks and difficulty](#-tracks-and-difficulty)
- [Configs](#%EF%B8%8F-configs)
- [Driving keys](#driving-keys)
- [Track seeds, training seeds and splits](#track-seeds-training-seeds-and-splits)

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

- `make dashboard` builds the dashboard from the final runs `final_s0` to `final_s2`, serves it on this computer and prints the local link (`http://localhost:8000/`), Ctrl+C stops it. It refuses to build when a final run is missing and prints the training command instead. `make dashboard SERVE=0` only builds into `runs/dashboard/`.

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

## 🚦 Commands

Every command runs from the repo folder. `make help` lists them all, and variables go after the command, like `make play SEED=1002`.

| Task | Command |
|---|---|
| Create the env and build the Rust core | `make setup` |
| Force a Rust rebuild, normally automatic | `make rust` |
| Run all tests, lint, auto-format | `make test`, `make lint`, `make format` |
| Env speed and track generation time | `make bench` |
| Drive a track by keyboard | `make play SEED=1002 DIFFICULTY=1` |
| Train one training seed of a config into `runs/<config>_s<seed>/` | `make train CONFIG=mvp SEED=0` |
| Train 3 training seeds at once on Windows or Linux, each run uses about one core | `make train-seeds CONFIG=final SEEDS="0 1 2"` |
| Watch training curves on http://localhost:6006 | `make tb` |
| Evaluate a baseline on validation tracks | `make eval POLICY=pure_pursuit N=50 DIFFICULTY=1` |
| Evaluate PPO, one zip per training seed | `make eval POLICY=ppo MODEL="runs/mvp_s0/best_model.zip runs/mvp_s1/best_model.zip"` |
| Final test table, once, never to tune | `make eval POLICY=ppo MODEL="..." SPLIT=test N=200` |
| Intervals and paired tests between evals | `make run CMD="python -m evaluation.stats ppo=runs/eval_ppo_val/eval.json pp=runs/eval_pure_pursuit_val/eval.json"` |
| Learning curve over training seeds | `make run CMD="python -m evaluation.plots curve --runs runs/mvp_s0 runs/mvp_s1 runs/mvp_s2 --label mvp"` |
| Bar chart and failure plots | `evaluation.plots bars --evals name=eval.json ...` and `evaluation.plots failures --policy ppo --model runs/mvp_s1/model.zip` |
| Watch a run drive 6 tracks, checkpoint after checkpoint | `make run CMD="python -m learning.watch --run runs/final_s2 --tracks 6"` |
| Side by side video into `videos/` | `make run CMD="python -m evaluation.demo --policies ppo pure_pursuit --model runs/mvp_s0/best_model.zip --seed 1000"` |
| Build the dashboard and serve it at a local link | `make dashboard`, or `make dashboard SERVE=0` to build only |
| Any command inside the env | `make run CMD="python x.py"` |

`make eval` defaults to the first 20 validation tracks at difficulty 0.5 and writes `runs/eval_<policy>_<split>/eval.json`. Pass `--out` through `make run CMD="python -m evaluation.evaluate ..."` to keep several difficulties apart.

Docker runs the same steps without installing anything. `make docker-build` once and after any change in `rust/` or the dependencies, `make docker-test`, `make docker-play`, `make docker-tb`, `make docker-shell` and `make docker-run CMD="python -m learning.train --config configs/mvp.yaml --seed 0"` for any other command.

## 🛣 Tracks and difficulty

`generate_track(seed, difficulty)` in `sim/track.py` builds every track, and the same track seed and difficulty always give the same one. It puts control points around a 70 m circle with some noise in angle and radius, draws a closed cubic spline through them and resamples it at one point per meter. Corners tighter than 12 m are widened in place so the inner barrier never folds, and loops whose far parts come too close are redrawn. The road is 8 m wide with 8 m of grass to the barrier on each side.

Difficulty is any number from 0 to 1. It sets the control points (6 to 12) and the noise (0.1 to 0.5), so higher means more and tighter corners.

| Difficulty | Control points | Points tighter than 25 m | Pure pursuit lap |
|---|---|---|---|
| 0 | 6 | 2% | 29.5 s |
| 0.5 | 9 | 10% | 32.3 s |
| 1 | 12 | 26% | 41.5 s |

Pick it with `DIFFICULTY=` in `make play` and `make eval`, or with `difficulty` in a config. Formulas are in [docs/PHYSICS.md](docs/PHYSICS.md).

## ⚙️ Configs

One YAML in `configs/` per experiment, and its file name is the run name. To try something, copy a config, change one thing and train it with 3 training seeds.

| Key | Meaning |
|---|---|
| `total_timesteps`, `n_envs` | Training length and parallel envs |
| `difficulty` | Track difficulty the agent trains on, the starting one with a curriculum |
| `curriculum.step`, `curriculum.threshold` | Optional. Raise the difficulty by `step` each time the validation lap rate reaches `threshold` |
| `curriculum.mix` | Optional. Share of training episodes whose difficulty is drawn uniformly below the current one |
| `reward.overspeed` | Optional. Penalty per step for driving faster than the car could still brake from for the corners within lidar range |
| `checkpoint_freq` | Steps between `checkpoints/*.zip` |
| `eval.freq`, `eval.tracks` | Validation every so many steps, on the first tracks from track seed 1000. Fills `val_curve.csv` and picks `best_model.zip` |
| `reward.*` | Weights of `RaceReward` in `learning/reward.py`, progress, time, jitter, grass, idle, crawl, off_road, stuck and lap |
| `ppo.*` | Passed straight to Stable-Baselines3 PPO |

| Config | What it is | Time on 8 cores |
|---|---|---|
| `tiny` | 20k steps, checks that everything runs | under a minute |
| `probe`, `probe_nojitter` | 200k step reward checks from week 2 | a few minutes |
| `probe_mvp`, `probe_ent`, `probe_crawl` | 300k step checks of `mvp`, with an entropy bonus and with crawl instead of idle | under 2 minutes |
| `mvp` | the first main run, 2M steps, one training seed of 3 never starts | about 5 minutes alone, 9 for 3 training seeds at once, ask the team first |
| `mvp_crawl` | `mvp` with crawl instead of idle | about 5 minutes |
| `mvp_curriculum` | `mvp_crawl` with the adaptive curriculum from difficulty 0 to 1 | about 5 minutes |
| `mvp_curriculum_4m` | `mvp_curriculum` with 4M steps | about 15 minutes for 3 training seeds at once |
| `exp_overspeed_4m` | `mvp_curriculum_4m` with the overspeed penalty | about 15 minutes for 3 training seeds at once |
| `final` | the frozen T11 config, `exp_overspeed_4m` with mixed difficulties, all 3 training seeds lap 92% or more at difficulty 1 | about 17 minutes for 3 training seeds at once |

## Driving keys

| Key | Action |
|---|---|
| Up | Accelerate, or brake while reversing |
| Down | Brake, and once stopped, reverse |
| Left, Right | Steer |
| Esc | Quit |

An attempt fails after 10 s in a row on grass, on reaching the barrier, after 10 s without progress, or after 120 s. A failed attempt or a finished lap puts you back on the start line. Corners taken too fast make the car drift, as [the physics guide](docs/PHYSICS.md) explains.

## Track seeds, training seeds and splits

| Split | Track seeds | Used for |
|---|---|---|
| Train | 100000 and up | Training only, random every episode |
| Validation | 1000 to 1999 | Everything while developing, `make eval` default |
| Test | 10000 and up | Final results only, with `SPLIT=test`. Never tune on it |

The evaluation script rejects training track seeds, so no training track reaches the results.

A **training seed** (0, 1 or 2, the `_s0` in run names) is a different thing. It fixes the starting weights of the network and the randomness of PPO, not a track. One run uses one training seed and thousands of track seeds, and the 3 runs of a config are 3 independent trainings of the same method.
