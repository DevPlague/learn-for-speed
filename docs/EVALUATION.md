# Evaluation

How S3 measures any driver on tracks it never trained on. Code lives in `evaluation/`, tests in `tests/test_baselines.py` and `tests/test_evaluate.py`.

## Track splits

Every track comes from a seed. The same seed and difficulty always give the same track.

| Split | Seeds | Used for |
|---|---|---|
| Train | 100000 and up | PPO training only. The env picks these when `track_seed` is None |
| Validation | 1000 to 1999 | Comparing runs, tuning, gates, debugging |
| Test | 10000 to 99999 | The final reported numbers, run once at the end |

`evaluate` raises an error for any training seed, so a training track can never reach the results. Never tune anything on the test split.

## What a baseline is

A baseline is a simple driver that needs no learning. It tells us what "good" and "bad" mean before PPO exists.

- **Random** (`RandomPolicy`) picks a uniform random action in [-1, 1] every step. It is the floor. If PPO does not beat it, PPO learned nothing.
- **Pure pursuit** (`PurePursuitPolicy`) is a classic hand-written controller. It cheats on purpose by reading the track from the env, which PPO never sees.
  - Steering aims at the centerline point 10 m ahead with the pure pursuit formula `atan(2 L sin(alpha) / d)`, where `L` is the wheelbase, `alpha` the angle to the point and `d` its distance.
  - Speed looks at points 10, 20 and 30 m ahead, estimates how sharp the coming corner is and picks the speed the tires can hold, `sqrt(GRIP / curvature)`, clipped between 6 and 16 m/s. Throttle or brake pushes the car toward that speed.

Pure pursuit is the reference that a good driver can lap these tracks. On validation seeds 1000 to 1004 it laps every track at difficulty 0, 0.5 and 1 in about 30 s. This was Gate 1 (the env is solvable).

Both baselines are callables, `action = policy(obs)`, exactly like a trained model, so the same evaluation code runs all of them.

## How an evaluation runs

`evaluate(policy, seeds, difficulty, env=None)` in `evaluation/evaluate.py`

1. For each seed it builds that track and puts the car on the start line.
2. It calls `policy(obs)` and `env.step(action)` until the episode ends with a lap, off road (10 s on grass or hitting the barrier), stuck (10 s without progress) or timeout (120 s).
3. It records the outcome of the episode and moves to the next seed.

Pure pursuit must read the same env it drives, so it is passed as `env=`. Other policies let `evaluate` create the env.

### Metrics

| Metric | Meaning |
|---|---|
| `lap_success_rate` | share of tracks where the car finished a lap |
| `off_track_rate` | share that ended off road |
| `stuck_rate` | share that ended stuck |
| `mean_lap_time` | seconds, over successful laps only, `-` when there are none |
| `mean_speed` | m/s, absolute, so reverse counts too |
| `mean_progress` | fraction of the lap reached, 0 to 1 |

The three failure rates do not have to add up to 1 minus the success rate, because a timeout counts as none of them. High stuck or off-track rates are also how we spot reward hacking, for example a car that learned to creep or spin.

## Evaluating a trained model

A PPO agent needs a model file saved by S2's training script, a `.zip` in `runs/<name>/`. Two cases

- **After training**, use the final model of the run.
- **During training**, use a checkpoint `.zip` the script saved on the way. This checks progress without stopping the run.

The model is loaded on CPU and always acts deterministically, so the same model on the same tracks gives the same numbers. `VecNormalize` only scales the reward, so the model gets raw observations and needs no extra stats file.

PPO results change with the training seed. Train the same config with 3 seeds and pass all 3 files. The table then shows mean ± std over them, which is what the report needs.

## Commands

Baselines, no model needed

```bash
make run CMD="python -m evaluation.evaluate --policy pure_pursuit --split val --n 20"
make run CMD="python -m evaluation.evaluate --policy random --split val --n 20"
```

One PPO model, or one checkpoint during training

```bash
make run CMD="python -m evaluation.evaluate --policy ppo --model runs/a_s0/model.zip --split val --n 20"
```

Three training seeds, mean ± std

```bash
make run CMD="python -m evaluation.evaluate --policy ppo --model runs/a_s0/model.zip runs/a_s1/model.zip runs/a_s2/model.zip --split val --n 50"
```

Final results only, once, at the end of W4

```bash
make run CMD="python -m evaluation.evaluate --policy ppo --model runs/a_s0/model.zip runs/a_s1/model.zip runs/a_s2/model.zip --split test --n 100 --out runs/final_ppo"
```

Options are listed in the README. Every run writes `runs/<out>/eval.json` with the arguments, the seeds and the metrics, and prints a Markdown table to paste into PLAN.md.

## Checklist before reporting a number

- Same `--n`, `--split` and `--difficulty` for every policy in one table
- Validation while developing, test only for the final table
- Three training seeds for PPO and ablations
- Off-track and stuck rates read, not only success
- Result logged in the PLAN.md experiment table
