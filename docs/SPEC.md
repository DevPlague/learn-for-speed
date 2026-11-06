# Spec

Technical contract of the project. Read it before touching a module interface. If an interface changes, update this file and the Decisions list in docs/PLAN.md in the same PR. Keep everything simple. Add structure only when a real need appears.

## System
A car drives on a procedurally generated closed track. The simulation core computes physics, sensors and track queries. A gymnasium environment wraps it and computes the reward. PPO trains on that environment. An evaluation module runs any policy on held-out tracks and produces the results table and the demos.

Data flow is `sim` then `learning` then `evaluation`. `evaluation` also uses `sim` directly, so every policy runs through the same code path.

## Structure and owners
| Folder | Owner | Responsibility |
|---|---|---|
| `sim/` | S1 (Simulation) | Track generator, gymnasium env, rendering, keyboard play |
| `rust/` | S1 (Simulation) | Per-step physics, car, lidar and track queries |
| `learning/` | S2 (Learning) | Reward, PPO training script, curriculum, checkpoints, logging |
| `evaluation/` | S3 (Evaluation) | Baselines, evaluation on held-out seeds, metrics, plots, videos |
| `configs/` | S2 | One YAML file per experiment |
| `tests/` | each owner tests their folder | pytest, run with `make test` |
| `docs/` | everyone | PLAN.md for planning and logs, SPEC.md for this contract, PHYSICS.md for the physics guide |

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
| `left_wall`, `right_wall` | NumPy arrays (N, 2) with the barriers, one road width of grass past each edge |
| `s` | NumPy array (N,) with the arclength of each centerline point |
| `heading` | NumPy array (N,) with the direction of travel in radians |
| `length` | Lap length in meters |

Points are about 1 m apart, width is 8 m. Progress and lateral offset come from the nearest centerline point, searched in a window around the previous index, then projected on its tangent (`locate` and `frenet` on the Rust core). `lookahead` in `sim/track.py` gives centerline points in the car frame, for baselines only.

### Environment
`RacingEnv(track_seed=None, difficulty=0.0, render_mode=None)` is a standard gymnasium env.

| Item | Contract |
|---|---|
| Observation | One float32 vector of `OBS_DIM` = 28 in [-1, 1]. Sensors only, 25 lidar, signed forward speed, steering, sideways slip speed (speed and slip both divided by 20 m/s). No track map data |
| Action | Float32 vector (2) in [-1, 1]. Steer (positive is left) and throttle. Negative throttle brakes while moving forward and reverses from standstill |
| `reset(seed=...)` | Car on the start line of a new track from `track_seed`, or random from the training range when None |
| `info` keys | `progress` (0 to 1 per lap), `lap_done`, `off_road`, `stuck`, `on_grass`, `speed` (forward only, it reads low during a drift), `lap_time` when the lap is done |
| Ends | Grass is allowed. Terminated on lap, off road (over 10 s in a row on grass or reaching the barrier) or stuck (under 2 m gained in 10 s). Truncated after 120 s. The next episode starts on the start line |
| `difficulty` | Plain attribute, the curriculum may change it between episodes (for example with `set_attr`) |

The observation size is exported as `OBS_DIM` in `sim/env.py` so learning and evaluation never hardcode it. Lidar has 25 rays over 360 degrees, denser in front, with a range of 30 m. Rays stop at road edges and barriers. Grass slows the car and lowers its grip. The env keeps a placeholder reward (`default_reward` in `sim/env.py`). Training replaces it with the `RaceReward` wrapper in `learning/`, so tuning the reward never touches `sim/`.

### Learning
| Item | Contract |
|---|---|
| `RaceReward(env, weights)` | Gymnasium wrapper in `learning/reward.py`. Reward is progress (laps) times `progress`, minus `time` per step, minus `jitter` times the steering change, minus `grass` per step on grass, minus `off_road` once, plus `lap` once. Weights come from the `reward` block of the config. Adds `is_success` (lap done) to `info` at the end of each episode |
| Training | `python -m learning.train --config configs/<name>.yaml --seed <n>` writes `runs/<name>_s<n>/` with `config.yaml`, `model.zip` (final model), `best_model.zip`, `checkpoints/`, `evaluations.npz`, `progress.csv` (SB3 logger, one row per PPO update) and TensorBoard events |
| Validation | Every `eval.freq` steps the policy drives each of the first `eval.tracks` validation seeds (from 1000) once with deterministic actions. `best_model.zip` is the best mean reward on them |
| `load_policy(path)` | In `learning/policy.py`. Returns a policy callable for any saved model. No normalization stats are needed because only the reward is normalized |

### Rust core
The crate in `rust/` builds the Python module `racing_core` with PyO3 and maturin. It is a uv path dependency, so `make setup` builds it and `uv run` rebuilds it when a Rust file changes. `make rust` forces a rebuild. Docker builds it inside the image, so changing `rust/` needs `make docker-build` again.

Track generation stays in Python. `make_core(track)` in `sim/track.py` passes the track arrays to `racing_core.TrackCore` once at reset. The core holds no car state. The env then calls it once per step.

`core.step(state, action, idx, on_grass) -> (state, hits, idx, frac, lateral)`

Arrays cross the boundary as float64 NumPy arrays. `core.cast`, `core.locate`, `core.frenet` and `racing_core.step_car` are there for tests and baselines. Constants are defined in Rust. Use them as `car.WHEELBASE` and `lidar.ANGLES` from `sim/car.py` and `sim/lidar.py`. The formulas are in docs/PHYSICS.md.

### Policy
A policy is any callable that maps an observation to an action. A baseline is a plain function or a small class with `__call__`. A trained model is wrapped in a small function that calls `model.predict` with deterministic actions and returns the action. Pure pursuit may need track data, it reads it from the env it is given. Useful attributes are `env.track`, `env.state` (x, y, yaw, v, steer, v_lat) and `env.idx` (nearest centerline index), with `lookahead` in `sim/track.py` to get points in the car frame. A trained agent never sees any of this.

### Evaluation
`evaluate(policy, seeds, difficulty) -> dict` runs one episode per seed with deterministic actions. The dict has lap success rate, off-track rate, stuck rate, mean lap time of successful laps, mean speed and mean progress. The same function evaluates PPO, random and pure pursuit. An optional `env` argument passes the env a policy reads from, like pure pursuit. Seeds must come from the validation or test split, training seeds raise an error.

### Config and runs
One YAML file per experiment in `configs/`. The training script copies the resolved config to `runs/<name>/` and writes checkpoints and logs there. Run names are `<config>_s<seed>`. Only models chosen on purpose are committed, as small files outside `runs/`.

## Technical decisions (decided)
| Topic | Decision |
|---|---|
| Language and tools | Python 3.12 only, Rust for the core, uv, make, Linux, Docker for headless runs |
| Simulation | Custom gymnasium env, bicycle model that grips in slow corners and drifts past the grip limit. Car state is [x, y, yaw, v, steer, v_lat] |
| Physics core | Rust for car step, lidar and centerline queries, with PyO3 and maturin. No NumPy fallback, Docker is the alternative to installing Rust |
| Tracks method | Periodic cubic spline through noisy points on a circle. Difficulty sets the point count (6 to 12) and the noise. Tracks with a corner tighter than the road or overlapping roads are redrawn |
| Car constants | dt 0.05 s, wheelbase 2.5 m, max steer 0.5 rad, max speed 20 m/s, reverse 5 m/s, grip limit 8 m/s^2 (4 on grass) |
| Tests | pytest finds the packages through `pythonpath = ["."]` in pyproject |
| Rendering | pygame, imported lazily, so training is headless |
| Tracks | Closed spline tracks, difficulty knob in [0, 1], curriculum |
| Observation | Sensors only, 25 lidar rays, speed, steering, slip, all scaled to [-1, 1] |
| Reward | Progress along the centerline, small time penalty, steering jitter penalty, off-road penalty, lap bonus |
| Learning | stable-baselines3 PPO with MlpPolicy, CPU torch, VecNormalize on reward |
| Seeds | Seed everything. Train from 100000, validation 1000 to 1999, test from 10000 |
| Config | YAML per experiment |
| Long runs | S2 launches them on a 10-core machine. `n_envs` in the config sets the parallel envs (`SubprocVecEnv`) |

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
| Reward weights and termination thresholds | S2 |
| Curriculum rule | S2 |
| Pure pursuit parameters | S3 |
| Number of evaluation tracks and difficulties | S3 |
