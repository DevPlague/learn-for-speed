# What S3 needs from S2

Evaluation (S3) runs every trained model on validation and test tracks and builds the results table, figures and videos. This page lists what the training side (S2) has to provide so that works. Points marked **proposal** are not agreed yet. Reply in the PR or at the weekly sync.

## Status on 9 Nov

Delivered by S2 and working with evaluation
- Run folders `runs/<config>_s<seed>/` with `model.zip`, `best_model.zip`, `config.yaml`, `checkpoints/`, `progress.csv` and `val_curve.csv` (PR #5)
- `val_curve.csv` feeds the learning curve figure directly

MVP results, 3 seeds of `configs/mvp.yaml`, validation 1000 to 1049, difficulty 0

| Seed | Laps | Off track | Stuck |
|---|---|---|---|
| 0 | 100% | 0% | 0% |
| 1 | 0% | 0% | 100% |
| 2 | 84% | 16% | 0% |
| mean | 61% ± 44 | 5% ± 8 | 33% ± 47 |

Gate 1 passes. The MVP target of 70% is not met on the mean because seed 1 creeps a few meters and stops on every track at 0% progress. Seed 2 fails off road in sharp corners taken too fast. Plots are in `runs/figures/failures_s1/` and `runs/figures/failures_s2/` on the S3 machine.

## What S3 needs now

1. **Fix seed 1.** A config where all 3 seeds learn. Idle and stuck penalties exist since T08 but seed 1 still ends stuck. Retrain the 3 mvp seeds after the fix and tell S3.
2. **T11.** 3 seeds of the target config, with the final difficulty the curriculum reaches.
3. **T12.** The ablation configs, 3 seeds each, same PPO seeds 0, 1, 2.
4. **Sharing.** `runs/` is not on GitHub. Share each run folder (`best_model.zip`, `model.zip`, `val_curve.csv`, `config.yaml`) on the shared drive, or tell S3 the config so S3 trains it locally at about 30 minutes per seed.
5. **Dates.** When T11 and T12 will be ready. S3 runs the test split once, before 22 Nov.

## 1. Observations and actions

`evaluate` feeds the model the raw env observation and sends its action straight to `env.step`.

- Keep `VecNormalize` on the **reward only** (`norm_obs=False`). If observations are normalized, tell S3 and save the stats file next to the model, because evaluation would then need to load it.
- Do not wrap the env in a way that changes observation or action shape (frame stacking, action repeat). If you need one, tell S3 first. It is an interface change for SPEC.md.
- Any change to obs, action or reward updates SPEC.md, as CLAUDE.md says.

## 2. Seeds

- Train with `track_seed=None`. The env then picks training tracks (100000 and up) by itself.
- Validation during training uses seeds 1000 to 1999. You can call our function directly

```python
from evaluation.evaluate import evaluate, split_seeds

evaluate(lambda obs: model.predict(obs, deterministic=True)[0], split_seeds("val", 20), 0.5)
```

- Never use test seeds (10000 and up). S3 runs those once for the final table.
- Training seeds of PPO itself (0, 1, 2) are separate from track seeds. Every reported config needs 3 of them.

## 3. Difficulty

`evaluate` takes one difficulty for the whole table, 0.5 by default.

- Tell S3 which difficulty the curriculum reaches at the end of training, so we evaluate at a level the agent was trained for, and also at 1.0 as a stress test.

## 4. Ablations (T12)

For each ablation, use one config name and 3 seeds, for example `no_lookahead_s0`, `no_lookahead_s1`, `no_lookahead_s2`. S3 evaluates them with the same command as the main model, so names are all we need.

## 5. Timeline, what S3 waits for

| When | S2 delivers | S3 then does |
|---|---|---|
| W2, by 8 Nov | first model or checkpoint (T07, T08), done | Gate 1, done, passes |
| W3, by 15 Nov | MVP model where all 3 seeds learn | MVP check, at least 70% laps on validation. First video done |
| W4, by 22 Nov | 3 seeds of the target config (T11) and the ablations (T12) | final test table, learning curves, failure analysis |
| W5, before 29 Nov | the model chosen for the demo, committed as a small file outside `runs/` | final videos and report figures |

## 6. Quick check S2 can run

After saving any model, this should print a table and not crash.

```bash
make run CMD="python -m evaluation.evaluate --policy ppo --model runs/<name>/model.zip --split val --n 5"
```

If it gets 0% laps while pure pursuit laps 100% on the same tracks, the model or the reward is the problem, not the env.

More detail on metrics and splits is in docs/EVALUATION.md.
