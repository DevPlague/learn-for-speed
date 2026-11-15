# Report draft, evaluation sections

Draft text for report sections 4 (Experiments) and 5 (Failure analysis). The results are the single **test** run (track seeds 10000 to 10199) of the frozen T11 config `configs/final.yaml`. Tables and figures are in `results/`.

## 4. Experiments

### Protocol

Every driver is scored on tracks it never trained on. Tracks come from track seeds. Training uses 100000 and up, validation 1000 to 1999 for development and model selection, test 10000 to 99999 once for the final numbers. The trained policy acts deterministically. An episode ends with a lap, off road (10 s off the road or a barrier hit), stuck (10 s without progress) or a 120 s timeout. We report lap success rate, off-track rate, stuck rate, lap time over successful laps, mean speed and mean progress. PPO is trained with 3 training seeds and reported per training seed and pooled. Every rate has a 95% interval and every comparison a paired test, McNemar for laps and Wilcoxon for progress and lap time, with Holm correction and a Friedman test across all policies. Section by section justification in docs/EVALUATION_STATISTICS.md.

### Baselines

- **Random** draws a uniform action every step. It is the floor.
- **Pure pursuit** is a hand-written controller that reads the track centerline, which PPO never sees. It steers toward a point 10 m ahead and picks the speed the tires can hold in the coming corner. It is the reference for a competent driver and shows the tracks are solvable (Gate 1).

### Final results, test 10000 to 10199

The T11 config `final` adds the adaptive curriculum, the overspeed penalty and mixed difficulties to the MVP, 4M steps per training seed. Each policy drove the same 200 test tracks once, at difficulty 0 and 1.

| Policy | Difficulty | Laps | 95% CI | Off track | Stuck | Lap time |
|---|---|---|---|---|---|---|
| PPO `final`, 3 training seeds | 0 | 100% | [1.00, 1.00] | 0% | 0% | 24.2 s |
| pure pursuit | 0 | 100% | [0.98, 1.00] | 0% | 0% | 29.5 s |
| random | 0 | 0% | [0.00, 0.02] | 0% | 100% | - |
| PPO `final`, 3 training seeds | 1 | 95.7% | [0.94, 0.97] | 3.2% | 1.2% | 36.3 s |
| pure pursuit | 1 | 100% | [0.98, 1.00] | 0% | 0% | 40.7 s |
| random | 1 | 0% | [0.00, 0.02] | 0% | 100% | - |

Friedman rejects equal performance at both difficulties (p < 0.0001). Pairwise, Holm corrected
- **PPO against random.** Every training seed laps more tracks (McNemar, 187 to 194 discordant tracks at difficulty 1, p < 0.0001).
- **PPO against pure pursuit.** At difficulty 0 there is no discordant track. At difficulty 1 pure pursuit laps 6, 7 and 13 tracks that training seeds 1, 0 and 2 miss and never the reverse (p 0.03, 0.03 and 0.0007). On tracks both finish PPO is 5.4 s faster at difficulty 0 and 4.4 s at difficulty 1 (Wilcoxon p < 0.0001).
- **Training seeds.** Per training seed the lap rate at difficulty 1 is 96.5%, 97% and 93.5%, and the t interval over training seeds is [0.91, 1.00]. The method now works for every training seed, not only for a lucky one.

Full output in `results/stats_test_d0.md` and `results/stats_test_d1.md`, bar charts `results/results_test_d0.png` and `results/results_test_d1.png`.

### How we got there, validation at difficulty 1

| Config | Training seed 0 | Training seed 1 | Training seed 2 |
|---|---|---|---|
| `mvp`, difficulty 0 only | 0% | 0% | 0% |
| `mvp_curriculum_4m`, curriculum | 100% | 100% | 0% |
| `exp_overspeed_4m`, + overspeed | 100% | 100% | 12% |
| `final`, + mixed difficulties | 100% | 98% | 88% |

- Training seed 2 of `mvp_curriculum_4m` never braked hard. It reached corners above their grip limit although the lidar showed them early enough to brake. Chained corners were not the cause.
- The overspeed term taught it to brake, but it braked so much that it rolled backwards into walls, since brake and reverse are the same pedal.
- Mixed difficulties removed the reversing and all 3 training seeds reached difficulty 1. On validation tracks 1050 to 1099, never used for model selection, they lap 98%, 96% and 92%.

### Learning curve

`results/curve_final.png`. Validation lap rate over the 3 training seeds. The dips follow each curriculum step, when validation moves to harder tracks, and the rate ends near 95% at difficulty 1.

## 5. Failure analysis

Test tracks at difficulty 1 with the T11 models, figures in `results/failures/`. Plots show the track, the barriers, the path colored by speed and a cross where the episode ended.

1. **Corner taken too fast (training seed 2, track 10065).** The car enters a 20 m corner 4.6 m/s above the allowed speed with 64% throttle and leaves the road on the outside at 58% of the lap. The overspeed term makes this rarer but does not remove it.
2. **Stops on the grass (training seed 2, track 10026).** The car brakes in a corner, slides onto the grass at 61% of the lap and stays there almost stopped, without throttle and without steering back, until the 10 s grass limit. It rarely saw recovery situations in training. Track 10057 fails the same way.
3. **Does not start on one track (training seed 0, track 10090).** The start line sits just before a tight 13 m corner. The car rolls two meters, stops with zero throttle and ends stuck at 1% of the lap. With the barrier close ahead the policy chooses to stand still. Track 10003 shows the same stop at 82% of the lap.

## Media

- Videos recorded with the MVP models, rerecord with `final` for the demo
- `videos/ppo_vs_pure_pursuit_1020.mp4` PPO next to pure pursuit, difficulty 0
- `videos/ppo_vs_pure_pursuit_1020_d1.mp4` same at difficulty 1
- `videos/lidar_s0_1021.mp4` single PPO view with lidar rays
- `videos/failure_s1_stuck_1000.mp4` and `videos/failure_s2_1008.mp4` failure cases
- Figures in `results/`, curve, bars at difficulty 0 and 1, failures
