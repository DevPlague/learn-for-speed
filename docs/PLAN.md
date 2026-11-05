# Plan

One flexible file: goal, scope, timeline, tasks, risks, logs. No sprints, no fixed owners. Dates assume 6 weeks, **Mon 26 Oct to Sun 6 Dec 2026**. The real deadline and format (report / slides / oral defense with demo) are **unknown**, so the plan is built to be *presentable at any time* from the end of week 3.

## Goal
Train a PPO agent that completes laps on **unseen procedurally generated tracks**, and show *why it works* (baselines, ablations, failure analysis) with clean demos.
This is **reinforcement learning** (PPO updates one network by gradients from reward), not an evolutionary algorithm (which breeds a population of networks without gradients).

## Scope tiers
| Tier | Contents | Done when |
|---|---|---|
| **MVP** (must ship) | Playable env, 3-5 fixed tracks, lidar + PPO, one reward, random + pure-pursuit baselines, one clean video, short report | Agent laps its tracks; tag `mvp` |
| **Target** | Procedural tracks, curriculum, held-out evaluation, 3 seeds, 3-4 ablations, polished demos and report | Results table + curves + videos |
| **Stretch** (at most one) | SAC vs PPO, static obstacles, dynamic tire model, ghost race vs earlier checkpoint, neuroevolution comparison | Only if Gate 2 is green |

## Timeline and gates
| Week | Focus | Exit criterion |
|---|---|---|
| **W1** 26 Oct - 1 Nov | Repo layout, `make setup` works for everyone, car physics, one track, rendering, drive by keyboard | `check_env` passes; everyone can run it |
| **W2** 2 - 8 Nov | Procedural tracks, lidar, reward, random + pure-pursuit baselines, **first PPO run** | **Gate 1:** pure pursuit laps (env solvable) and PPO beats random |
| **W3** 9 - 15 Nov | Iterate reward/obs, curriculum, parallel envs | **MVP tag:** >= 70% laps on validation tracks. A demo exists from here |
| **W4** 16 - 22 Nov | Held-out evaluation, 3 seeds, ablations | **Gate 2:** choose a stretch goal or cut it |
| **W5** 23 - 29 Nov | Stretch (optional), best videos, report draft | **Code freeze** Sun 29 Nov |
| **W6** 30 Nov - 6 Dec | Report, slides, demo rehearsal. No new features | Buffer for surprises and the oral defense |

If the deadline is earlier, cut from the bottom: Stretch, then ablations, then seeds (keep >= 2). The MVP survives any cut.

## Oral defense / demo buffer
- From W3: a committed working model, 2-3 pre-rendered videos on a shared drive and a USB stick, slides with offline-playable clips.
- W6 is rehearsal: a live demo **and** a no-computer fallback (videos on a phone, slides as PDF).
- Everyone can explain every part; rotate who presents which part in rehearsals.
- 5-minute script: task (30 s) / how: obs, action, reward (60 s) / live agent on unseen tracks (90 s) / results table + ablations (60 s) / a failure case (30 s) / limits (30 s).
- Likely questions: why RL vs hand controller? why PPO, not evolution? how do you know it generalizes (held-out seeds)? how could the reward be exploited? why do seeds differ?

## How we work
- Take a task by adding your initials; keep at most 2 in Doing each. Nobody is assigned. Letters in brackets (S1 Simulation, S2 Learning, S3 Evaluation) only suggest an area.
- 20-minute weekly sync (tasks, gate, risks); async message when blocked.
- `main` always runs; short branches; one teammate reviews each PR.
- One config per experiment; tiny run before long jobs; log every run below.
- Done = works via `make`, tests/lint pass, logged here, someone else looked at it.

## Tasks
**Doing:** -  |  **Review:** T17 (S1)

**Backlog**
- [ ] T01 Everyone runs `make setup`; first person commits `uv.lock`; GitHub repo + protected `main`
- [ ] T17 (S1) Rust core crate, binding choice, `make rust` target and Dockerfile step. Profile a NumPy version first
- [ ] T06 (S3) Baselines: random and pure pursuit; pure pursuit must lap tracks (Gate 1)
- [ ] T07 (S2) Training script: SB3 PPO, parallel envs, config file per run, validation tracks, checkpoints, TensorBoard
- [ ] T08 (S2) First MVP run (fixed gentle tracks); watch it; fix env/reward bugs
- [ ] T09 (S2) Curriculum + procedural tracks; check it reaches full difficulty
- [ ] T10 (S3) Evaluation script (test seeds, table, JSON, mean +- std over seeds)
- [ ] T11 (S2) Final runs: 3 seeds of the target config
- [ ] T12 (S2) Ablations (3 seeds each): no lookahead (obsolete since D17, S2 to pick a replacement), no curriculum, one of {obs normalization, no smoothness penalty, steer-only}
- [ ] T13 (S3) Learning-curve figure; failure analysis on 3 test tracks
- [ ] T14 (S3) Videos: side by side vs pure pursuit, one failure case, lidar view; trajectory figure
- [ ] T15 Gate 2 decision (stretch or not) logged below
- [ ] T16 Report draft, slides, two demo rehearsals (live and offline fallback)

**Done**
- [x] T02 Code layout in CLAUDE.md and docs/SPEC.md (2026-11-02)
- [x] T03 (S1) Env: car, spline tracks with difficulty, rendering, keyboard play (2026-11-05)
- [x] T04 (S1) Env: lidar observation, placeholder reward, termination (2026-11-05)
- [x] T05 (S1) Tests: geometry, physics, `check_env`, reproducibility with seeds. S3 may add more (2026-11-05)

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| First PPO runs fail from an env/reward/obs bug, not PPO | High (70-80%) | Tests, pure-pursuit sanity check, watch videos early (Gate 1 in W2) |
| Learns one track but not unseen ones | Medium (25-35%) | Procedural tracks from the start, curriculum, held-out validation |
| Laps but slow or wobbly | Medium (~60%) | Accept for MVP; tune time/smoothness penalties; speed is stretch |
| Training too slow | Medium | Headless sim, parallel envs, CPU torch, tiny run first, fewer seeds |
| Someone cannot run it locally | Low | Docker path, outputs bind-mounted |
| Deadline/format changes | Unknown | Demo-ready from W3, scope cuts above |
| Reward hacking (spinning, creeping, hugging walls) | Medium | `debug-reward` skill, videos, off-track/stuck rates in every eval |
| Work concentrates on one person | Medium | Rotate who runs experiments; everyone reviews PRs |

Probabilities are judgement-based estimates, not measurements. Ending with nothing usable: under 5%.

**Fallback ladder (if learning stalls):** 1) steer-only, fixed speed. 2) easier/fixed tracks. 3) richer obs or denser reward. 4) lower speed cap or wider road. 5) pivot to a documented failure analysis (still earns marks).

## Experiment log (test seeds only)
| ID | Config | Seeds | Change vs reference | Test success / lap [s] | Conclusion |
|---|---|---|---|---|---|
| E00 | baselines | - | random, pure pursuit | | |
| E01 | mvp | 1 | fixed gentle tracks, no curriculum | | |
| E02 | target | 3 | procedural + curriculum (reference) | | |

## Decisions
- 2026-10-25 D01 Python 3.12. Why: every dependency (torch, pygame, scipy, SB3) ships wheels for it.
- 2026-10-25 D02 Custom gymnasium env, NumPy physics, pygame only to render. Why: fast headless training.
- 2026-10-25 D03 Vector observations, no pixels. Why: MLP trains fast on CPU; a CNN is ~10x slower and riskier.
- 2026-10-25 D04 PPO, not evolution. Why: sample-efficient, standard, explainable via reward design.
- 2026-10-25 D05 Bicycle model + grip limit. Why: forces braking for corners, still simple to debug.
- 2026-10-25 D06 Procedural spline tracks + curriculum. Why: unlimited tracks give a real generalization test.
- 2026-10-25 D07 CPU-only PyTorch via the PyTorch CPU index. Why: PPO + small MLP is CPU-bound; avoids ~3 GB of CUDA libs.
- 2026-10-25 D08 Seeds: train >= 100000, validation 1000-1999, test 10000+. Why: no leakage into reported numbers.

- 2026-11-02 D09 Code in subpackages by role: sim/ (S1), learning/ (S2), evaluation/ (S3). Why: clear ownership with three people in parallel.
- 2026-11-02 D10 `.claude/` stays gitignored. Why: each student has personal skills. Shared knowledge lives in CLAUDE.md, SPEC.md and PLAN.md.
- 2026-11-02 D11 No strict code style beyond ruff. Why: keep code simple, avoid friction.
- 2026-11-05 D12 `evaluate(policy, seeds, difficulty, env=None)` takes an optional env and rejects training seeds. Why: pure pursuit reads the same env it drives, tests pass a mock env, and no training track can leak into results.
- 2026-11-02 D12 Physics core in Rust, one call per step, built with a separate `make rust` target. Replaces the NumPy physics part of D02. Why: simulation speed, S1 chooses the binding.
- 2026-11-02 D13 Contracts are plain functions and NumPy arrays, config is one YAML per experiment, PRs need one review. Why: simplest option for a small team.
- 2026-11-02 D14 Sim is pure NumPy first, Rust only after profiling, binding PyO3 with maturin. Refines D12. Why: unblocks S2 and S3 for Gate 1 and Rust is not installed yet.
- 2026-11-02 D15 Observation is 23 floats (11 lidar, speed, steering, heading error, lateral offset, 4 lookahead points at 5, 10, 20, 30 m). Action is [steer, throttle/brake]. Env ships a placeholder reward until S2 tunes it. Why: first usable env for the team.
- 2026-11-02 D16 Tracks are periodic cubic splines through noisy circle points, difficulty sets point count and noise, bad tracks are redrawn. Why: simple, deterministic per seed, valid by construction.
- 2026-11-02 D17 Observation is sensors only, 25 lidar rays over 360 degrees plus speed and steering (27 floats). Replaces the obs part of D15. Lookahead, heading error and lateral offset are removed, so the "no lookahead" ablation in T12 no longer applies. Why: the agent must drive only from what its lidar sees.
- 2026-11-02 D18 Car can reverse. Grass is allowed but slow and slippery with a small penalty. Barriers stand one road width past each edge. An attempt fails after 3 s on grass, at the barrier, or after 10 s without progress, and the next episode starts on the start line. Why: closer to real driving and the agent can recover from mistakes.
- 2026-11-02 D19 Car drifts. The nose may turn harder than the tires hold (14 vs 8 m/s^2), the excess becomes sideways slip that the tires pull back at the grip limit. Slip speed is added to the observation (28 floats). Grass limit raised to 10 s. Refines D05, D17 and D18. Why: realistic fast cornering, and the agent must sense when it slides.
- 2026-11-05 D20 Rust with PyO3 and maturin replaces NumPy for the car step, lidar and track queries. Replaces D14. Track generation stays in Python, so every seed keeps its track. Rust is required locally, with Docker as the alternative. Measured env.step went from about 190 us to about 30 us, about 5k to about 30k steps/s per core. Track generation still costs 13 to 24 ms per reset. Why: simulation speed, the student chose to do it now.
- 2026-11-05 D21 Physics constants live in Rust and are exported to Python. Why: one source of truth next to the equations. Tuning needs a rebuild, which uv does by itself.
- 2026-11-05 D22 Three physics fixes after the exact port. Body-frame update instead of rotating the velocity, so a gripping corner loses only drag. Speed clipped again after that update. Slip observation scale raised from 10 to 20 m/s, since a spin reaches 19 m/s. Why: found by measurement, and the last one changes the observation.
- 2026-11-05 D23 Physics guide in docs/PHYSICS.md, with its numbers checked by tests/test_sim_physics.py. Why: everyone must be able to explain the physics in the oral defense.

## Report skeleton
1. Problem and why RL  2. Environment (physics, tracks, sensors, reward)  3. Method (PPO, curriculum, hyperparameters)  4. Experiments (baselines, held-out results, ablations)  5. Failure analysis  6. Conclusions and limits.
