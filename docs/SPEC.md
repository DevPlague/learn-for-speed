# Spec

Technical contract of the project. Read it before touching a module interface. If an interface changes, update this file and the Decisions list in docs/PLAN.md in the same PR. Keep everything simple. Add structure only when a real need appears.

## System
A car drives on a procedurally generated closed track. The simulation core computes physics, sensors and track queries. A gymnasium environment wraps it and computes the reward. PPO trains on that environment. An evaluation module runs any policy on held-out tracks and produces the results table and the demos.

Data flow is `sim` then `learning` then `evaluation`. `evaluation` also uses `sim` directly, so every policy runs through the same code path.

## Structure and owners
| Folder | Owner | Responsibility |
|---|---|---|
| `sim/` | S1 (Simulation) | Track generator, car physics, lidar, gymnasium env, rendering. Physics core in Rust |
| `learning/` | S2 (Learning) | Reward, PPO training script, curriculum, checkpoints, logging |
| `evaluation/` | S3 (Evaluation) | Baselines, evaluation on held-out seeds, metrics, plots, videos |
| `configs/` | S2 | One YAML file per experiment |
| `tests/` | each owner tests their folder | pytest, run with `make test` |
| `docs/` | everyone | PLAN.md for planning and logs, SPEC.md for this contract |

Proposed file names are track, car, lidar, env, render in `sim/`, reward and train in `learning/`, baselines, evaluate and demo in `evaluation/`. Owners may merge, split or rename files. Only the contracts below must stay stable.

Only the owner edits a folder. Anyone else asks the owner or opens a PR the owner reviews.

## Contracts
There are no web endpoints. Modules talk through Python function calls with NumPy arrays and plain dicts. No abstract classes and no formal interfaces.

### Track
Built by `generate_track(seed, difficulty) -> Track`. Seed splits are in CLAUDE.md. Difficulty is a float in [0, 1].

A `Track` is a small dataclass with these fields.
| Field | Meaning |
|---|---|
| `centerline` | NumPy array (N, 2), closed loop, meters |
| `width` | Road width in meters |
| `left`, `right` | NumPy arrays (N, 2) with the road edges |

How the core finds progress and lateral offset from a position is an open question owned by S1.

### Environment
`RacingEnv(track_seed=None, difficulty=0.0, render_mode=None)` is a standard gymnasium env.

| Item | Contract |
|---|---|
| Observation | One float32 vector in [-1, 1]. Lidar rays, then speed, steering, heading error, lateral offset, centerline lookahead points |
| Action | Float32 vector (2) in [-1, 1]. Steer and throttle or brake |
| `reset(seed=...)` | New track from `track_seed`, or random from the training range when None |
| `info` keys | `progress` (0 to 1 per lap), `lap_done`, `off_road`, `stuck`, `lap_time` when done |
| Ends | Terminated on lap, off road or stuck. Truncated on timeout |

The observation size is exported as a constant so learning and evaluation never hardcode it.

### Rust core
Python calls Rust once per env step. The call takes the action and the current state and returns the new state, the lidar readings and the progress data. Arrays cross the boundary as NumPy arrays. Track generation stays in Python and the track arrays are passed to Rust at reset. The binding choice and the build details are open questions owned by S1. The build target is `make rust`, to be added by S1 together with the Dockerfile step.

### Policy
A policy is any callable that maps an observation to an action. A baseline is a plain function or a small class with `__call__`. A trained model is wrapped in a small function that calls `model.predict` with deterministic actions and returns the action. Pure pursuit may need track data, it reads it from the env it is given.

### Evaluation
`evaluate(policy, seeds, difficulty) -> dict` runs one episode per seed with deterministic actions. The dict has lap success rate, off-track rate, stuck rate, mean lap time of successful laps, mean speed and mean progress. The same function evaluates PPO, random and pure pursuit.

### Config and runs
One YAML file per experiment in `configs/`. The training script copies the resolved config to `runs/<name>/` and writes checkpoints and logs there. Run names are `<config>_s<seed>`. Only models chosen on purpose are committed, as small files outside `runs/`.

## Technical decisions (decided)
| Topic | Decision |
|---|---|
| Language and tools | Python 3.12 only, uv, make, Linux, Docker for headless runs |
| Simulation | Custom gymnasium env, kinematic bicycle model with a grip limit |
| Physics core | Rust for car step, lidar and centerline queries |
| Rendering | pygame, imported lazily, so training is headless |
| Tracks | Closed spline tracks, difficulty knob in [0, 1], curriculum |
| Observation | About 11 lidar rays, speed, steering, heading error, lateral offset, a few lookahead points, all scaled to [-1, 1] |
| Reward | Progress along the centerline, small time penalty, steering jitter penalty, off-road penalty, lap bonus |
| Learning | stable-baselines3 PPO with MlpPolicy, CPU torch, VecNormalize on reward |
| Seeds | Seed everything. Train from 100000, validation 1000 to 1999, test from 10000 |
| Config | YAML per experiment |

## Code style
Ruff for lint and format through `make lint` and `make format`. No other rules. Write short readable code, comments only where the reason is not obvious. English only. Do not use semicolons or colons in comments and docs prose.

## Git flow
- One branch per task, named `<student>/<task>`, for example `s1/track-gen`.
- Small PRs into `main`, one teammate reviews, `make test` and `make lint` pass.
- `main` always runs.
- Never commit `runs/`, `videos/`, `*.pkl` or large files.
- Claim a task in docs/PLAN.md with your initials before you start it.

## Open questions
Nothing here is decided. The owner decides, then moves the answer to the tables above and logs it in docs/PLAN.md.

| Question | Owner |
|---|---|
| Rust binding (PyO3 with maturin or another) and how the crate is laid out | S1 |
| Does Rust really pay off, profile a NumPy version first | S1 |
| How Rust finds progress and lateral offset on the track | S1 |
| Exact lidar count, range and number of lookahead points | S1 |
| Track generation method and how difficulty maps to shape | S1 |
| Reward weights and termination thresholds | S2 |
| Curriculum rule and parallel env count | S2 |
| Who launches long training runs and where | S2 |
| Pure pursuit parameters | S3 |
| Number of evaluation tracks and difficulties | S3 |
| How pytest finds the packages (a `pythonpath` setting in pyproject) | S1 |
