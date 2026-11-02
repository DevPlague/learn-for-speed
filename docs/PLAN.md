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
- Take a task by adding your initials; keep at most 2 in Doing each. Nobody is assigned.
- 20-minute weekly sync (tasks, gate, risks); async message when blocked.
- `main` always runs; short branches; one teammate reviews each PR.
- One config per experiment; tiny run before long jobs; log every run below.
- Done = works via `make`, tests/lint pass, logged here, someone else looked at it.

## Tasks
**Doing:** -  |  **Review:** -

**Backlog**
- [ ] T01 Everyone runs `make setup`; first person commits `uv.lock`; GitHub repo + protected `main`
- [ ] T02 Decide code layout; document it in CLAUDE.md (<= 8 lines)
- [ ] T03 Env: car (bicycle + grip limit), spline tracks with difficulty, rendering, keyboard play
- [ ] T04 Env: lidar + lookahead observation, reward, termination (off-road, stuck, timeout, lap)
- [ ] T05 Tests: geometry, physics, `check_env`, reproducibility with seeds
- [ ] T06 Baselines: random and pure pursuit; pure pursuit must lap tracks (Gate 1)
- [ ] T07 Training script: SB3 PPO, parallel envs, config file per run, validation tracks, checkpoints, TensorBoard
- [ ] T08 First MVP run (fixed gentle tracks); watch it; fix env/reward bugs
- [ ] T09 Curriculum + procedural tracks; check it reaches full difficulty
- [ ] T10 Evaluation script (test seeds, table, JSON, mean +- std over seeds)
- [ ] T11 Final runs: 3 seeds of the target config
- [ ] T12 Ablations (3 seeds each): no lookahead, no curriculum, one of {obs normalization, no smoothness penalty, steer-only}
- [ ] T13 Learning-curve figure; failure analysis on 3 test tracks
- [ ] T14 Videos: side by side vs pure pursuit, one failure case, lidar view; trajectory figure
- [ ] T15 Gate 2 decision (stretch or not) logged below
- [ ] T16 Report draft, slides, two demo rehearsals (live and offline fallback)

**Done:** -

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

## Report skeleton
1. Problem and why RL  2. Environment (physics, tracks, sensors, reward)  3. Method (PPO, curriculum, hyperparameters)  4. Experiments (baselines, held-out results, ablations)  5. Failure analysis  6. Conclusions and limits.
