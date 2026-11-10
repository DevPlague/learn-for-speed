# S3 handoff for the next agent

Read this before working on `evaluation/`. It says what S3 (Evaluation) already has, what is left after the T13 and T14 pull request is merged, and the traps found so far. CLAUDE.md, docs/SPEC.md and docs/PLAN.md still rule. Commands and metrics are explained in docs/EVALUATION.md.

## Ground rules that matter most here

- Work on branch `s3/evaluation`, never on main. Never commit or push. Give the student the `git add` and `git commit` block, following the commit format in CLAUDE.md.
- Edit only `evaluation/`, `tests/test_baselines.py`, `tests/test_evaluate.py`, `tests/test_plots.py`, `tests/test_demo.py` and S3 docs. `learning/` and `dashboard/` belong to S2, `sim/` and `rust/` to S1.
- Validation seeds 1000 to 1999 while developing. Test seeds 10000 and up only once, for the final table. Training seeds are rejected by `check_seeds`.
- Ask the student before any training longer than 10 minutes. One 2M step `mvp` seed takes about 30 minutes on the student's 8 cores.
- Do not read raw logs or TensorBoard files. Print short summaries.
- No colons or semicolons in comments and docs prose.

## What exists

| Piece | File | Task |
|---|---|---|
| Random and pure pursuit baselines | `evaluation/baselines.py` | T06, done |
| `evaluate`, `run_episode`, `split_seeds`, `check_seeds`, `aggregate`, `to_markdown`, `make_policy`, `ppo_policy`, CLI | `evaluation/evaluate.py` | T10, done |
| Learning curve from `val_curve.csv`, results bar chart, trajectory plot, failure search, CLI `curve`, `bars`, `failures` | `evaluation/plots.py` | T13, code done |
| Side by side demo videos with lidar | `evaluation/demo.py` | T14, code done |
| Confidence intervals and paired tests from eval.json | `evaluation/stats.py`, theory in `docs/EVALUATION_STATISTICS.md` | done (D30) |
| Process guide, S2 handoff | `docs/EVALUATION.md`, `docs/FOR_S2.md` | done |

S2's training (`learning/train.py`) calls our `evaluate` during training and writes `runs/<config>_s<seed>/` with `model.zip`, `best_model.zip`, `val_curve.csv`, `progress.csv`, `checkpoints/` and `config.yaml`. Nothing in `runs/` or `videos/` is on GitHub, so runs must be trained locally or copied from a teammate.

## Results so far

Fill this table from the student's latest printed summaries and keep it in sync with PLAN.md E00 and E01.

| Policy | Tracks | Laps | Off track | Stuck | Lap time |
|---|---|---|---|---|---|
| pure pursuit | val 1000 to 1004, difficulty 0.5 | 100% | 0% | 0% | 31.0 s |
| random | val 1000 to 1004, difficulty 0.5 | 0% | 0% | 100% | - |
| pure pursuit | val 1000 to 1049, difficulty 0 | 100% | 0% | 0% | 29.5 s |
| random | val 1000 to 1049, difficulty 0 | 0% | 0% | 100% | - |
| ppo mvp, seeds 0 1 2 | val 1000 to 1049, difficulty 0 | 61% ± 44 | 5% ± 8 | 33% ± 47 | 24.1 s |

Per seed the PPO lap rate is 100%, 0% and 84%. Seed 1 is stuck on every track at 0% progress. Seed 2 fails off road in tight corners taken too fast (seeds 1005, 1008). On seeds 1020 to 1049 the numbers are 100%, 0% and 87%, so selection bias is small. At difficulty 0.5 and 1 PPO falls to 49% ± 35 laps with more off road, pure pursuit stays at 100%. Draft report text for these results and the three failures is in `docs/REPORT_EVALUATION.md`. Eval files are `runs/eval_*_val_d0/eval.json`, figures in `runs/figures/`, videos `videos/ppo_vs_pure_pursuit_1020.mp4` and `videos/failure_s2_1008.mp4`.

## Next steps after the T13 and T14 PR is merged

1. **Gate 1 and MVP check.** Evaluate the three `mvp` seeds on validation and compare with the baselines on the same tracks and difficulty (0, the MVP trains at 0). Gate 1 passes if PPO beats random. MVP passes at 70% laps or more. Log both in PLAN.md and tell the student to tag `mvp` if it passes.
2. **Selection bias.** `best_model.zip` is picked on validation seeds 1000 to 1019, the first `eval.tracks`. Numbers on those seeds are optimistic. Report validation on seeds the model was not selected on, or rely on test for the final table. A clean fix is a `start` offset in `split_seeds`, which is an interface change, so ask the student and update SPEC.md and PLAN.md.
3. **Failure analysis (T13).** Run `plots failures` on the weakest seed, keep three plots, and help the student write one or two lines per failure. S2 reported that one seed learned to reverse slowly until stuck, which is a good case to show.
4. **Videos (T14).** Record PPO next to pure pursuit on a validation track, one failure, and a single PPO view for the lidar. Names of the policies are not drawn on the frames yet, so a small label overlay is a possible improvement. Ask before adding it.
5. **Final results (W4, by 22 Nov).** After S2 finishes T11 (3 seeds of the target config) and T12 (ablations), run `evaluate --split test --n 200` once for PPO, each ablation and both baselines with the same `--n` and difficulty. Run `evaluation.stats` on all of them, build the bar chart, copy the small PNGs and the Markdown table to `results/` and commit only those.
6. **Report and slides (T16).** Results section, learning curve, bar chart, failure plots and videos. Help the student rehearse the likely questions in PLAN.md.

## Known issues outside S3

- PR #6 (S2 dashboard) was merged into `s2/mvp`, not into main, because it was stacked on PR #5. S2 must open a pull request from `s2/mvp` to main. Until then `dashboard/` is not on main.
- `docs/PLAN.md` Decisions had two D12 entries. The S3 one is now D29. D28 comes with the dashboard PR.

## Quick health check

```bash
make test
make lint
make run CMD="python -m evaluation.evaluate --policy pure_pursuit --split val --n 5"
```

Pure pursuit must lap every track. If it does not, the env changed and evaluation numbers are not comparable with earlier ones.
