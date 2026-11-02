# racing-rl

2D top-down racing car. PPO (stable-baselines3) learns to drive procedurally generated tracks; judged on held-out tracks vs baselines (random, pure pursuit) and ablations. School project, 3 students, ~6 weeks. Python 3.12 only, uv, Linux. Deadline/format unknown: keep a demo-ready state.

## Commands (use make, never raw pip/python)
- `make setup` env; `make run CMD="python x.py"`; `make test | lint | format | tb | clean`
- Docker (headless, outputs bind-mounted): `make docker-build | docker-run CMD=".." | docker-shell | docker-test | docker-tb`

## Layout
- `sim/` S1 (Simulation): track, car, lidar, env, render. Physics core in Rust (`make rust`, to be added by S1).
- `learning/` S2 (Learning): reward, PPO training, curriculum. `configs/` one YAML per experiment.
- `evaluation/` S3 (Evaluation): baselines, evaluate, plots, videos. `tests/` each owner tests their folder.
- Only the owner edits a folder. Contracts between modules: docs/SPEC.md.
- Plan, tasks, risks, experiment and decision logs: docs/PLAN.md.

## Design (intended)
- Custom gymnasium env; physics core in Rust (one call per step, NumPy arrays), pygame only for rendering (lazy import) so training is headless.
- Kinematic bicycle car with grip limit; closed spline tracks with a difficulty knob in [0,1]; curriculum.
- Obs: ~11 lidar rays + speed, steering, heading error, lateral offset, a few centerline lookahead points (all scaled to [-1,1]). Action: [steer, throttle/brake] in [-1,1].
- Reward: progress along centerline - small time penalty - steering jitter; off-road penalty; lap bonus. PPO MlpPolicy, CPU torch, VecNormalize on reward.

## Rules
- Seed everything. One config file per experiment; save it next to the run outputs (runs/<name>/).
- Track seeds: random training >= 100000, validation 1000-1999, test 10000+. Never tune on test; report test only.
- Never commit runs/, videos/, *.pkl or large files. Commit only deliberate final models and small result tables/figures.
- Ask before launching any training > 10 min. Check changes with a tiny run first.
- Do not read raw logs, tensorboard files or uv.lock. Read short printed summaries only.
- Changing obs/action/reward: log it in docs/PLAN.md (Decisions) and keep tests passing.
- Read docs/SPEC.md before touching a module interface. Interface changes update SPEC.md and PLAN.md Decisions in the same PR.
- Never decide architecture, libraries or interfaces alone. Ask the student first, keep code simple, no over-engineering.
- Work on a branch, never on main. Never run git commit or push and never add yourself to commits. Give the student a code block with the git add and git commit commands.
- No semicolons or colons in comments and docs prose.
- `.claude/` is gitignored on purpose (personal skills). Shared rules live only in CLAUDE.md, SPEC.md and PLAN.md.
- Keep answers and diffs short. English only (code, comments, docs).

## Skills (load on demand)
train-experiment, evaluate-agent, debug-reward, record-demo, update-docs
