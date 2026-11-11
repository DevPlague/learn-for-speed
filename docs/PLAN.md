# Plan

One flexible file: goal, scope, timeline, tasks, risks, logs. No sprints, no fixed owners. Dates assume 6 weeks, **Mon 26 Oct to Sun 6 Dec 2026**. The real deadline and format (report / slides / oral defense with demo) are **unknown**, so the plan is built to be *presentable at any time* from the end of week 3.

## Goal
Train a PPO agent that completes laps on **unseen procedurally generated tracks**, and show *why it works* (baselines, statistics, failure analysis) with clean demos.
This is **reinforcement learning** (PPO updates one network by gradients from reward), not an evolutionary algorithm (which breeds a population of networks without gradients).

## Scope tiers
| Tier | Contents | Done when |
|---|---|---|
| **MVP** (must ship) | Playable env, 3-5 fixed tracks, lidar + PPO, one reward, random + pure-pursuit baselines, one clean video, short report | Agent laps its tracks, tag `mvp` |
| **Target** | Procedural tracks, curriculum, held-out evaluation, 3 training seeds, polished demos and report | Results table + curves + videos |
| **Stretch** (at most one) | SAC vs PPO, static obstacles, dynamic tire model, ghost race vs earlier checkpoint, neuroevolution comparison | Only if Gate 2 is green |

## Timeline and gates
| Week | Focus | Exit criterion |
|---|---|---|
| **W1** 26 Oct - 1 Nov | Repo layout, `make setup` works for everyone, car physics, one track, rendering, drive by keyboard | `check_env` passes, everyone can run it |
| **W2** 2 - 8 Nov | Procedural tracks, lidar, reward, random + pure-pursuit baselines, **first PPO run** | **Gate 1:** pure pursuit laps (env solvable) and PPO beats random |
| **W3** 9 - 15 Nov | Iterate reward/obs, curriculum, parallel envs | **MVP tag:** >= 70% laps on validation tracks. A demo exists from here |
| **W4** 16 - 22 Nov | Held-out evaluation, 3 training seeds | **Gate 2:** choose a stretch goal or cut it |
| **W5** 23 - 29 Nov | Stretch (optional), best videos, report draft | **Code freeze** Sun 29 Nov |
| **W6** 30 Nov - 6 Dec | Report, slides, demo rehearsal. No new features | Buffer for surprises and the oral defense |

If the deadline is earlier, cut from the bottom: Stretch, then training seeds (keep >= 2). The MVP survives any cut.

## Oral defense / demo buffer
- From W3: a committed working model, 2-3 pre-rendered videos on a shared drive and a USB stick, slides with offline-playable clips.
- W6 is rehearsal: a live demo **and** a no-computer fallback (videos on a phone, slides as PDF).
- Everyone can explain every part, rotate who presents which part in rehearsals.
- 5-minute script: task (30 s) / how: obs, action, reward (60 s) / live agent on unseen tracks (90 s) / results table with intervals (60 s) / a failure case (30 s) / limits (30 s).
- Likely questions: why RL vs hand controller? why PPO, not evolution? how do you know it generalizes (held-out track seeds)? how could the reward be exploited? why do training seeds differ?

## How we work
- Take a task by adding your initials, keep at most 2 in Doing each. Nobody is assigned. Letters in brackets (S1 Simulation, S2 Learning, S3 Evaluation) only suggest an area.
- 20-minute weekly sync (tasks, gate, risks), async message when blocked.
- `main` always runs, short branches, one teammate reviews each PR.
- One config per experiment, tiny run before long jobs.
- Done = works via `make`, tests/lint pass, someone else looked at it.

## Tasks
**Doing:** T09 (S2)  |  **Review:** none

**Backlog**
- [ ] T09 (S2) Curriculum + procedural tracks, check it reaches full difficulty. Adaptive curriculum in `ValCurveCallback` and `configs/mvp_curriculum.yaml`. At 4M steps (`mvp_curriculum_4m`) 2 of 3 training seeds reach difficulty 1 and lap 100% of validation tracks 1000 to 1049 there, training seed 2 stays at 0.25 and fails by leaving the road
- [ ] T11 (S2) Final runs: 3 training seeds of the curriculum config
- [ ] T13 (S3) Failure analysis on 3 test tracks with the T11 model (curve and validation analysis done)
- [ ] T15 Gate 2 decision (stretch or not)
- [ ] T16 Report draft, slides, two demo rehearsals (live and offline fallback)

**Final model.** The config is frozen when all 3 training seeds learn, none under 70% laps on validation track seeds 1050 and up (never used for model selection), and the curriculum reaches difficulty 1. Only then the test split runs, once, before 22 Nov. Code freeze is 29 Nov.

**Done**
- [x] T02 Code layout in CLAUDE.md and docs/SPEC.md (2026-11-02)
- [x] T03 (S1) Env: car, spline tracks with difficulty, rendering, keyboard play (2026-11-05)
- [x] T04 (S1) Env: lidar observation, placeholder reward, termination (2026-11-05)
- [x] T01 Everyone runs `make setup`, first person commits `uv.lock`, GitHub repo + protected `main` (2026-11-05)
- [x] T17 (S1) Rust core crate, binding choice, `make rust` target and Dockerfile step. Profile a NumPy version first (2026-11-05)
- [x] T06 (S3) Baselines: random and pure pursuit, pure pursuit must lap tracks (Gate 1) (2026-11-05)
- [x] T10 (S3) Evaluation script (test track seeds, table, JSON, mean +- std over training seeds) (2026-11-05)
- [x] T05 (S1) Tests: geometry, physics, `check_env`, reproducibility with track seeds. S3 may add more (2026-11-05)
- [x] T07 (S2) Training script: SB3 PPO, parallel envs, config file per run, validation tracks, checkpoints, TensorBoard (2026-11-06)
- [x] T08 (S2) First MVP run on gentle tracks, 3 training seeds, evaluated on validation. Gate 1 passes, MVP 70% not met because one training seed never starts (2026-11-09)
- [x] T18 (S2) Local dashboard with training curves, validation and test tables, synced replay grid and physics plots, built from runs/ (2026-11-09)
- [x] T14 (S3) Videos side by side vs pure pursuit, failure case, lidar view, trajectory figure (2026-11-10)

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| First PPO runs fail from an env/reward/obs bug, not PPO | High (70-80%) | Tests, pure-pursuit sanity check, watch videos early (Gate 1 in W2) |
| Learns one track but not unseen ones | Medium (25-35%) | Procedural tracks from the start, curriculum, held-out validation |
| Laps but slow or wobbly | Medium (~60%) | Accept for MVP, tune time/smoothness penalties, speed is stretch |
| Training too slow | Medium | Headless sim, parallel envs, CPU torch, tiny run first, fewer training seeds |
| Someone cannot run it locally | Low | Docker path, outputs bind-mounted |
| Deadline/format changes | Unknown | Demo-ready from W3, scope cuts above |
| Reward hacking (spinning, creeping, hugging walls) | Medium | `debug-reward` skill, videos, off-track/stuck rates in every eval |
| Work concentrates on one person | Medium | Rotate who runs experiments, everyone reviews PRs |

Probabilities are judgement-based estimates, not measurements. Ending with nothing usable: under 5%.
