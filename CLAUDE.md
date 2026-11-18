# learn-for-speed

2D top-down racing car. PPO (stable-baselines3) learns to drive procedurally generated tracks; judged on held-out tracks vs baselines (random, pure pursuit). School project, 3 students, ~6 weeks. Python 3.12 only, uv, Linux. Deadline/format unknown: keep a demo-ready state.

## Commands (use make, never raw pip/python)
- `make setup` env; `make play` drive by keyboard; `make run CMD="python x.py"`; `make train CONFIG=tiny SEED=0`; `make eval POLICY=pure_pursuit` (val split); `make test | lint | format | rust | bench | tb | dashboard | clean`
- Docker (headless, outputs bind-mounted): `make docker-build | docker-run CMD=".." | docker-shell | docker-test | docker-play | docker-tb`

## Layout
- `sim/` S1 (Simulation): track generation, env, render, play. `rust/` S1: per-step physics. Formulas in docs/PHYSICS.md.
- `learning/` S2 (Learning): reward, PPO training, curriculum. `configs/` one YAML per experiment.
- `evaluation/` S3 (Evaluation): baselines, evaluate, plots, videos. `tests/` each owner tests their folder.
- `dashboard/` S2, local web dashboard built from runs/ into runs/dashboard/ (`make dashboard RUNS="mvp_s0"`).
- Only the owner edits a folder. Contracts between modules: docs/SPEC.md.
- Plan, tasks and risks: docs/PLAN.md.

## Design (intended)
- Custom gymnasium env; physics core in Rust (one call per step, NumPy arrays), pygame only for rendering (lazy import) so training is headless.
- Bicycle car that grips in slow corners and drifts in fast ones, can reverse; closed spline tracks with a difficulty knob in [0,1]; curriculum.
- Obs: sensors only, 25 lidar rays + speed, steering, slip (all scaled to [-1,1]). No track map. Action: [steer, throttle/brake] in [-1,1].
- Reward: progress along centerline - small time penalty - steering jitter; off-road penalty; lap bonus. PPO MlpPolicy, CPU torch, VecNormalize on reward.

## Rules
- Seed everything. One config file per experiment; save it next to the run outputs (runs/<name>/).
- Track seeds: random training >= 100000, validation 1000-1999, test 10000+. Never tune on test; report test only.
- Always write track seed (which track) or training seed (PPO run seed 0, 1, 2, the `_s0` in run names), never a bare seed.
- Commit only the light run files (config.yaml, progress.csv, val_curve.csv, eval.json, best_model.zip and runs/figures), .gitignore keeps the rest of runs/ out. Never commit videos/, *.pkl or other large files.
- Ask before launching any training > 10 min. Check changes with a tiny run first.
- Do not read raw logs, tensorboard files or uv.lock. Read short printed summaries only.
- Changing obs/action/reward: update docs/SPEC.md and keep tests passing.
- Read docs/SPEC.md before touching a module interface. Interface changes update SPEC.md in the same PR.
- Never decide architecture, libraries or interfaces alone. Ask the student first, keep code simple, no over-engineering.
- Work on a branch, never on main. Never run git commit or push and never add yourself to commits. Give the student a code block with the git add and git commit commands.
- No semicolons or colons in comments and docs prose. Commit messages never contain semicolons.
- Commit title is `type(scope): summary`, for example `fix(app): tester feedback, import limits and rotation slider`. Types are feat, fix, refactor, test, docs, chore. If the work matches a task in docs/PLAN.md, the summary starts with its ID and wording (`feat(learning): T07 training script with PPO`). Otherwise write a short summary of what was done.
- Commit body groups the changes under short headings (config, code, tests, docs). Each heading has bullets with the concrete facts, naming files, functions and key figures.
- `.claude/` is gitignored on purpose (personal skills). Shared rules live only in CLAUDE.md, SPEC.md and PLAN.md.
- Keep answers and diffs short. English only (code, comments, docs).

## Skills (load on demand)
train-experiment, evaluate-agent, debug-reward, record-demo, update-docs
