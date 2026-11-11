# Report draft, evaluation sections

Draft text for report sections 4 (Experiments) and 5 (Failure analysis). Numbers are on **validation** tracks and only show progress. The final report replaces every table with the single **test** run (seeds 10000 and up) once T11 and T12 exist. Placeholders are marked TODO.

## 4. Experiments

### Protocol

Every driver is scored on tracks it never trained on. Tracks come from seeds. Training uses 100000 and up, validation 1000 to 1999 for development and model selection, test 10000 to 99999 once for the final numbers. The trained policy acts deterministically. An episode ends with a lap, off road (10 s off the road or a barrier hit), stuck (10 s without progress) or a 120 s timeout. We report lap success rate, off-track rate, stuck rate, lap time over successful laps, mean speed and mean progress. PPO is trained with 3 seeds and reported per seed and pooled. Every rate has a 95% interval and every comparison a paired test, McNemar for laps and Wilcoxon for progress and lap time, with Holm correction and a Friedman test across all policies. Section by section justification in docs/EVALUATION_STATISTICS.md.

### Baselines

- **Random** draws a uniform action every step. It is the floor.
- **Pure pursuit** is a hand-written controller that reads the track centerline, which PPO never sees. It steers toward a point 10 m ahead and picks the speed the tires can hold in the coming corner. It is the reference for a competent driver and shows the tracks are solvable (Gate 1).

### MVP results, validation 1000 to 1049

| Policy | Difficulty | Laps | Off track | Stuck | Lap time |
|---|---|---|---|---|---|
| pure pursuit | 0 | 100% | 0% | 0% | 29.5 s |
| random | 0 | 0% | 0% | 100% | - |
| PPO mvp, 3 seeds | 0 | 61% ± 44 | 5% ± 8 | 33% ± 47 | 24.1 s |
| pure pursuit | 1 | 100% | 0% | 0% | 31.3 s |
| random | 1 | 0% | 0% | 100% | - |
| PPO mvp, 3 seeds | 1 | 49% ± 35 | 17% ± 12 | 34% ± 47 | 26.1 s |

Per seed at difficulty 0 the lap rate is 100%, 0% and 84%. At difficulty 1 it is 78%, 0% and 70%.

Reading
- PPO beats random clearly, so the agent learns to drive from lidar alone (Gate 1).
- When PPO laps, it is about 5 s faster than pure pursuit, because pure pursuit uses a conservative speed limit.
- The mean hides a split. Two seeds drive well, one never learns to move (seed 1). The large std is that seed, not noise around the mean. This is why we train 3 seeds.
- The MVP trains only on gentle tracks (difficulty 0). On harder tracks it loses about 10 points of lap rate and the failures move from stuck to off road.
- Difficulty 0.5 and 1 give almost the same numbers (49% both). The difficulty knob may change the tracks less than expected. To be checked with S1.
- `best_model` is selected on validation 1000 to 1019. On 1020 to 1049 alone the lap rates are 100%, 0% and 87%, so selection bias is small.

### Intervals and tests on validation

| Difficulty | Policy | Lap rate | 95% CI over tracks |
|---|---|---|---|
| 0 | PPO mvp pooled | 0.61 | [0.58, 0.65] |
| 0 | pure pursuit | 1.00 | [0.93, 1.00] |
| 0 | random | 0.00 | [0.00, 0.07] |
| 1 | PPO mvp pooled | 0.49 | [0.43, 0.56] |
| 1 | pure pursuit | 1.00 | [0.93, 1.00] |

Friedman rejects equal performance at both difficulties (p < 0.0001). With Holm correction, PPO seeds 0 and 2 lap more tracks than random (McNemar p < 0.0001) and seed 1 does not differ from random. At difficulty 1 every PPO seed laps fewer tracks than pure pursuit (p ≤ 0.002), while on tracks both finish PPO is about 5 s faster (Wilcoxon p < 0.0001). The t interval over the 3 seeds covers the whole range [0, 1], so the training method itself is not yet reliable.

### Final results

TODO after T11 and T12. Test split, 200 tracks for all, `evaluation.stats` output with intervals and Holm corrected tests, PPO target config 3 seeds, ablations 3 seeds each, both baselines. Bar chart `results/results_test.png` and the Markdown table.

### Learning curve

`runs/figures/curve_mvp.png`. Validation lap rate rises from about 30% to about 58% by 0.8M steps and then flattens. The wide band is seed 1 staying at 0. TODO replace with the T11 curve.

## 5. Failure analysis

Plots show the track, the barriers, the path colored by speed and a cross where the episode ended.

1. **Never starts (seed 1, track 1000, any track).** The car creeps a few meters at under 5 m/s and stops with 0% progress until the stuck timeout. The policy found a local optimum where it avoids the off-road penalty by not driving. The idle and stuck penalties from T08 were not enough for this seed. Seed 0 shows the same freeze on one hard track (1010 at difficulty 1, speed 0 the whole episode), so the behavior is not unique to seed 1.
2. **Corner taken too fast (seed 0, track 1005, difficulty 1).** The car runs a long straight at the 20 m/s top speed and reaches a sharp left kink without braking. It goes straight across onto the inside and off the road at 42% of the lap. The lidar sees the corner only at short range, so it brakes too late.
3. **Drift off at the end of the lap (seed 2, track 1005, difficulty 0).** After a fast lap the car exits the last corner wide, slows to about 8 m/s and stays on the grass beside the road until the off-road timeout at 98% progress. It does not steer back once it is off, which suggests it rarely saw recovery situations in training.

TODO redo these three on test tracks with the T11 model, as T13 asks.

## Media

- `videos/ppo_vs_pure_pursuit_1020.mp4` PPO (25.1 s) next to pure pursuit (30.2 s), difficulty 0
- `videos/ppo_vs_pure_pursuit_1020_d1.mp4` same at difficulty 1, 26.1 s vs 30.5 s
- `videos/lidar_s0_1021.mp4` single PPO view with lidar rays
- `videos/failure_s1_stuck_1000.mp4` and `videos/failure_s2_1008.mp4` failure cases
- Figures in `runs/figures/`, curve, bars at difficulty 0 and 1, failures
