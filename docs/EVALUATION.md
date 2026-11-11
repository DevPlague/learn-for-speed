# Evaluation

How S3 measures any driver on tracks it never trained on. Code lives in `evaluation/`, tests in `tests/test_baselines.py` and `tests/test_evaluate.py`.

## Track splits

Every track comes from a track seed. The same track seed and difficulty always give the same track.

| Split | Track seeds | Used for |
|---|---|---|
| Train | 100000 and up | PPO training only. The env picks these when `track_seed` is None |
| Validation | 1000 to 1999 | Comparing runs, tuning, gates, debugging |
| Test | 10000 to 99999 | The final reported numbers, run once at the end |

`evaluate` raises an error for any training track seed, so a training track can never reach the results. Never tune anything on the test split.

## What a baseline is

A baseline is a simple driver that needs no learning. It tells us what "good" and "bad" mean before PPO exists.

- **Random** (`RandomPolicy`) picks a uniform random action in [-1, 1] every step. It is the floor. If PPO does not beat it, PPO learned nothing.
- **Pure pursuit** (`PurePursuitPolicy`) is a classic hand-written controller. It cheats on purpose by reading the track from the env, which PPO never sees.
  - Steering aims at the centerline point 10 m ahead with the pure pursuit formula `atan(2 L sin(alpha) / d)`, where `L` is the wheelbase, `alpha` the angle to the point and `d` its distance.
  - Speed looks at points 10, 20 and 30 m ahead, estimates how sharp the coming corner is and picks the speed the tires can hold, `sqrt(GRIP / curvature)`, clipped between 6 and 16 m/s. Throttle or brake pushes the car toward that speed.

Pure pursuit is the reference that a good driver can lap these tracks. On validation track seeds 1000 to 1049 it laps every track at difficulty 0, 0.5 and 1, in 29.5 s, 32.3 s and 41.5 s on average. This was Gate 1 (the env is solvable).

Both baselines are callables, `action = policy(obs)`, exactly like a trained model, so the same evaluation code runs all of them.

## How an evaluation runs

`evaluate(policy, seeds, difficulty, env=None)` in `evaluation/evaluate.py`

1. For each track seed it builds that track and puts the car on the start line.
2. It calls `policy(obs)` and `env.step(action)` until the episode ends with a lap, off road (10 s on grass or hitting the barrier), stuck (10 s without progress) or timeout (120 s).
3. It records the outcome of the episode and moves to the next track seed.

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

PPO results change with the training seed. Train the same config with 3 training seeds and pass all 3 files. The table then shows mean ± std over them, which is what the report needs.

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
make run CMD="python -m evaluation.evaluate --policy ppo --model runs/a_s0/model.zip runs/a_s1/model.zip runs/a_s2/model.zip --split test --n 200 --out runs/final_ppo"
```

Options are listed in the README. Every run writes `runs/<out>/eval.json` with the arguments, the track seeds and the metrics, and prints a Markdown table.

## Report figures (T13)

`evaluation/plots.py` makes static PNG figures for the report and slides. They go to `runs/figures/` and only the final ones are copied to `results/` and committed.

**Learning curve.** Reads `val_curve.csv`, which training writes every `eval.freq` steps with the metrics above on the validation tracks. It plots lap success rate and mean progress against training steps, the mean over training seeds with a ± std band. Training seeds are cut to the checks they all reached.

```bash
make run CMD="python -m evaluation.plots curve --runs runs/mvp_s0 runs/mvp_s1 runs/mvp_s2 --label mvp --out runs/figures/curve_mvp.png"
```

**Failure analysis.** Drives a policy on the tracks of a split, keeps the first episodes that did not finish a lap and plots each one. The plot shows the road edges, the barriers (red dotted), the path colored by speed and a red cross where it ended, titled with the track seed, the reason and the progress.

```bash
make run CMD="python -m evaluation.plots failures --policy ppo --model runs/mvp_s1/model.zip --split val --n 50 --count 3"
```

For each failure write one or two lines in the report on why it happened, for example the speed before the cross shows a corner taken too fast, a short path ending on the start line shows a car that never got going. Use validation tracks while developing and the test split once for the final three.

**Results bar chart.** Lap, off-track and stuck rates per policy from `eval.json` files, with std bars when PPO has several training seeds. Each entry is `name=path`.

```bash
make run CMD="python -m evaluation.plots bars --evals ppo=runs/eval_ppo_val/eval.json pure_pursuit=runs/eval_pure_pursuit_val/eval.json random=runs/eval_random_val/eval.json"
```

## Demo videos (T14)

`evaluation/demo.py` lets several policies drive the same track at once and joins their views left to right in one MP4 at 20 fps. Each view is labeled at the bottom with the policy name and, once that car is done, its outcome. The view is S1's renderer with the lidar rays, the minimap and the speed, lap and time display. A car that finishes keeps its last frame until the others are done. Videos go to `videos/`, which is never committed. Copy the good ones to the shared drive.

```bash
make run CMD="python -m evaluation.demo --policies ppo pure_pursuit --model runs/mvp_s0/best_model.zip --seed 1000"
make run CMD="python -m evaluation.demo --policies ppo --model runs/mvp_s1/best_model.zip --seed 1003 --out videos/failure.mp4"
```

Pick the failing track seed from the `failures` command above.

## Confidence intervals and tests

`evaluation/stats.py` reads two or more `eval.json` files run on the same tracks and prints each policy with a 95% interval, then paired tests between every pair with Holm corrected p-values, and a Friedman test when there are three or more. The theory and the choice of each test are in docs/EVALUATION_STATISTICS.md.

```bash
make run CMD="python -m evaluation.stats ppo=runs/eval_ppo_val_d0/eval.json pure_pursuit=runs/eval_pure_pursuit_val_d0/eval.json random=runs/eval_random_val_d0/eval.json --out runs/stats_val_d0.md"
```

Files written before this change have no per-track episodes, rerun `evaluate` for them.

## Checklist before reporting a number

- Same `--n`, `--split` and `--difficulty` for every policy in one table
- Validation while developing, test only for the final table
- Three training seeds for every PPO config
- Off-track and stuck rates read, not only success
- 95% interval and paired test p-values from `evaluation.stats` next to every comparison
- Result added to docs/REPORT_EVALUATION.md
