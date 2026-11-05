# What S3 needs from S2

Evaluation (S3) runs every trained model on validation and test tracks and builds the results table, figures and videos. This page lists what the training side (S2) has to provide so that works. Points marked **proposal** are not agreed yet. Reply in the PR or at the weekly sync and we log the decision in PLAN.md.

## 1. Model files

- Save with `model.save(...)` from stable-baselines3 PPO, so the file is a `.zip` that `PPO.load` opens.
- One folder per run, `runs/<config>_s<seed>/`, as SPEC.md says. Final model at `runs/<config>_s<seed>/model.zip`.
- The resolved config copied next to it, so we know what each model is.
- **proposal** Checkpoints during training as `runs/<config>_s<seed>/checkpoints/*.zip` (for example with `CheckpointCallback`). Then S3 can evaluate a run before it ends and catch problems early.

## 2. Observations and actions

`evaluate` feeds the model the raw env observation and sends its action straight to `env.step`.

- Keep `VecNormalize` on the **reward only** (`norm_obs=False`). If observations are normalized, tell S3 and save the stats file next to the model, because evaluation would then need to load it.
- Do not wrap the env in a way that changes observation or action shape (frame stacking, action repeat). If you need one, tell S3 first. It is an interface change for SPEC.md.
- Any change to obs, action or reward goes into PLAN.md Decisions, as CLAUDE.md says.

## 3. Seeds

- Train with `track_seed=None`. The env then picks training tracks (100000 and up) by itself.
- Validation during training uses seeds 1000 to 1999. You can call our function directly

```python
from evaluation.evaluate import evaluate, split_seeds

evaluate(lambda obs: model.predict(obs, deterministic=True)[0], split_seeds("val", 20), 0.5)
```

- Never use test seeds (10000 and up). S3 runs those once for the final table.
- Training seeds of PPO itself (0, 1, 2) are separate from track seeds. Every reported config needs 3 of them.

## 4. Learning curve

The report needs a learning-curve figure with mean ± std over 3 seeds (T13). S3 must not read raw TensorBoard files.

- **proposal** Write `runs/<config>_s<seed>/progress.csv` with the SB3 logger (`configure(folder, ["csv", "tensorboard"])`). It has columns like `time/total_timesteps` and `rollout/ep_rew_mean`.
- **proposal** Even better, every N steps run the `evaluate` call above on 20 validation seeds and append one row to `runs/<config>_s<seed>/val_curve.csv` with columns `timesteps,lap_success_rate,mean_progress`. Success rate is far easier to explain than reward.

## 5. Difficulty

`evaluate` takes one difficulty for the whole table, 0.5 by default.

- Tell S3 which difficulty the curriculum reaches at the end of training, so we evaluate at a level the agent was trained for, and also at 1.0 as a stress test.

## 6. Ablations (T12)

For each ablation, use one config name and 3 seeds, for example `no_lookahead_s0`, `no_lookahead_s1`, `no_lookahead_s2`. S3 evaluates them with the same command as the main model, so names are all we need.

## 7. Timeline, what S3 waits for

| When | S2 delivers | S3 then does |
|---|---|---|
| W2, by 8 Nov | first model or checkpoint (T07, T08) | Gate 1, PPO vs random on 20 validation tracks |
| W3, by 15 Nov | MVP model | MVP check, at least 70% laps on validation, first demo video |
| W4, by 22 Nov | 3 seeds of the target config (T11) and the ablations (T12) | final test table, learning curves, failure analysis |
| W5, before 29 Nov | the model chosen for the demo, committed as a small file outside `runs/` | final videos and report figures |

## 8. Quick check S2 can run

After saving any model, this should print a table and not crash.

```bash
make run CMD="python -m evaluation.evaluate --policy ppo --model runs/<name>/model.zip --split val --n 5"
```

If it gets 0% laps while pure pursuit laps 100% on the same tracks, the model or the reward is the problem, not the env.

More detail on metrics and splits is in docs/EVALUATION.md.
