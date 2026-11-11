# Evaluation and model comparison, the theory behind it

This document explains how S3 measures and compares drivers and why, following Tema 8 of Extensiones de Machine Learning (UPCT 2026/2027, "Evaluación de Modelos. Selección de Métricas"). Each section names the slides it applies. Commands are in docs/EVALUATION.md, the code in `evaluation/evaluate.py` and `evaluation/stats.py`.

## 1. Why a careful evaluation (Tema 8 section 1)

The unit opens with the idea that a model is only as good as the way it is measured. Metrics are not neutral, they decide which approach looks "better" (slide 5). For us the risk is concrete. A reinforcement learning agent optimizes a reward, not our metrics, and it can learn behaviours that score well on a naive number while driving badly. An example from our own runs is training seed 1, which learned to stay still. Its mean speed is not zero and nothing crashes, yet it never completes a lap.

The goals of our evaluation are the ones on slide 4.

- Verify real performance, on tracks the agent never trained on
- Compare models fairly, PPO against baselines and against its ablations
- Support decisions, for example Gate 1, the MVP check and which model goes to the demo

## 2. The problem and what a sample is

Before choosing metrics, Metrics Reloaded (slide 25) asks for the "fingerprint" of the problem. Ours is not image classification or segmentation, so we translate the idea.

- **Task.** Sequential control. A policy drives a car around a closed track until it finishes a lap or fails.
- **Sample unit.** One **episode on one track**. Each track comes from a seed, so the same seed gives the same track for every policy.
- **Data splits.** Training tracks 100000 and up, validation 1000 to 1999, test 10000 to 99999. Test is run once for the final table. This avoids the most common evaluation error, measuring on data used for tuning (slide 16, distinct train and test distribution).
- **Paired design.** All policies drive the **same** list of track seeds. The results of two policies on track 1007 are paired observations of the same "item", like two classifiers predicting on the same dataset (slide 40, "mismo dataset, pareado"). This drives every test choice below.
- **Two sources of randomness.** The track (which tracks we sampled) and the training seed (PPO trained three times with seeds 0, 1, 2 gives three different agents). Both have to appear in the uncertainty we report.

## 3. Metric selection (Tema 8 sections 1 to 3)

### 3.1 The metrics

| Metric | Type | What it answers |
|---|---|---|
| `lap_success_rate` | proportion of tracks | Does the car finish a lap. The main metric |
| `off_track_rate` | proportion | How often it fails by leaving the road or hitting a barrier |
| `stuck_rate` | proportion | How often it fails by not moving forward for 10 s |
| `mean_progress` | mean in [0, 1] | How far around the lap it got, partial credit for failures |
| `mean_lap_time` | seconds, successful laps only | How fast it drives when it succeeds |
| `mean_speed` | m/s | Context for the others |

Each episode ends in exactly one outcome, lap, off road, stuck or timeout. The first three rates are therefore the shares of a multi-class outcome, like the rows of a confusion matrix with one true class "should finish a lap".

### 3.2 Why not a single number

Slides 7 and 8 show how a global accuracy can hide total failure, a spam filter that never discriminates still scores 97%. The same trap exists here.

- Lap rate alone does not say **why** the car fails. 40% failures all stuck (the car learned not to drive) and 40% all off road (it drives too fast) call for different fixes. That is why the three outcome rates are always reported together, a combination of complementary metrics (slide 46).
- Mean reward is deliberately **not** an evaluation metric. It is what the agent optimizes, so it is the number most likely to look good while behaviour is bad, and its scale changes whenever the reward is edited.
- Progress gives a graded signal where lap rate is all or nothing, so two failing agents can still be told apart.

### 3.3 Prevalence and difficulty (slide 9)

Slide 9 shows that accuracy is not comparable across datasets with different prevalence. Our equivalent of prevalence is the **difficulty** of the tracks. A lap rate at difficulty 0 is not comparable with one at difficulty 1. Every table therefore fixes one difficulty and one set of track seeds, and a comparison is only made between rows that share both.

### 3.4 Unequal cost of errors (slides 10, 11)

Slides 10 and 11 discuss errors that do not cost the same and introduce Expected Cost. In our case going off road and getting stuck are both failures, but they mean different things. We do not merge them into one weighted cost, because there is no agreed cost ratio between them. Keeping them as separate rates shows the trade-off directly, which is what a cost-based metric would summarize.

### 3.5 Lap time and missing values (slide 23)

Slide 23 shows that how missing values are treated can change a mean from 0.90 to 0.60. Lap time is only defined when a lap was completed, so it is "missing" on every failed track. If we simply averaged over successful laps, a policy that only finishes the easy tracks would look fast. The rules are

- `mean_lap_time` in the tables is reported together with the lap rate, never alone
- when **comparing** lap time between two policies, only tracks where **both** finished a lap are used, so both are timed on the same tracks

## 4. Aggregation (slides 20, 21)

Slide 21 warns that a global mean can hide a group where the model fails. For PPO the groups are the training seeds. The MVP result shows why this matters.

| Seed | Lap rate |
|---|---|
| 0 | 100% |
| 1 | 0% |
| 2 | 84% |
| mean | 61% |

The mean of 61% describes none of the three agents. It averages two working agents with one that never learned. The rules are

- per-seed values are **always** shown next to the mean
- the std over seeds (± in the tables) shows how much training luck matters, it is not an error bar of the estimate

## 5. Uncertainty, confidence intervals (slides 20, 46)

Slide 20 lists "no mostrar intervalos de confianza ni desviaciones" as an interpretation error, and slide 46 asks for them in every comparison. A lap rate of 0.84 on 50 tracks could easily be 0.75 or 0.92 on another 50 tracks. The interval says how far.

We use three intervals, each matched to the kind of number and to a source of randomness. All are 95%.

### 5.1 Wilson interval for a rate of one model

A lap rate is a proportion, k laps out of n tracks. The textbook normal interval p ± 1.96 sqrt(p(1-p)/n) breaks at the edges. At 100% laps it gives a width of zero, as if the result were certain, and near 0 it can go negative. The **Wilson score interval** fixes both, stays inside [0, 1] and works for small n.

Example from validation, pure pursuit lapped 50 of 50 tracks. The Wilson interval is [0.93, 1.00]. With 50 tracks we can say the true lap rate on this kind of track is at least 93%, not that it is exactly 100%.

### 5.2 Bootstrap interval over tracks

For progress, and for the PPO lap rate pooled over its seeds, the per-track values are not a simple proportion. We use the **percentile bootstrap**. Resample the 50 tracks with replacement 10000 times, compute the mean each time, and take the 2.5% and 97.5% percentiles. It assumes no distribution, which suits bounded metrics like progress (slide 40 notes that bounded metrics break symmetry assumptions).

### 5.3 t interval over training seeds

The two intervals above treat the trained agents as fixed and only ask "what if we had driven other tracks". They do not say what happens if we **train again**. For that the sample is the training seed, and we have three. The Student t interval with 2 degrees of freedom gives, for the MVP lap rate, 0.61 with seed interval [0, 1] after clipping. It is honest. Three agents that disagree this much tell us almost nothing about the next one. The report states this limit instead of hiding it.

The two kinds of interval answer two different questions.

| Interval | Randomness covered | Question |
|---|---|---|
| Wilson or bootstrap over tracks | which tracks were drawn | how good are **these** trained agents on new tracks |
| t over seeds | which training run we got | how good is **the training method** |

## 6. Hypothesis tests (Tema 8 section 4)

### 6.1 The framework (slides 36 to 39)

A comparison between two policies A and B is a statistical experiment.

- **H0** there is no real difference between A and B on these tracks
- **H1** there is a difference
- **Significance level** α = 0.05, the accepted probability of a type I error, claiming a difference that is not there (slide 38)
- **p-value**, the probability of a difference at least as large as observed if H0 were true. If p < α we reject H0 (slide 39)

Two cautions from slide 37 apply to every result below. A non-significant test means "no evidence of a difference with these tracks", not "they are equal". A significant difference does not mean a large one, so the effect size (median difference, discordant counts) is always printed next to the p-value.

### 6.2 Choosing the test (slide 40)

The test follows from the data, as slide 40 insists.

| Data | Paired | Distribution | Test |
|---|---|---|---|
| lapped yes or no per track | yes, same tracks | binary | **McNemar**, exact form |
| progress per track | yes | bounded, skewed, many ties at 0 and 1 | **Wilcoxon signed-rank** |
| lap time on tracks both finished | yes | skewed, outliers | **Wilcoxon signed-rank** |
| three or more policies | yes | ranks | **Friedman** then pairwise with correction |

Tests we do **not** use and why.

- **Paired t-test** (slides 40, 41). It assumes the differences are normal. Progress differences pile up at exactly 0 (both lapped) and at large values (one got stuck), far from normal. Slide 41 itself warns that with few pairs and a bounded metric the t-test is unreliable.
- **Mann-Whitney U** (slide 40). It is for **independent** samples, models evaluated on different datasets. Our policies share tracks, so the paired Wilcoxon is the right one and uses more information.
- **DeLong** (slides 44, 45). It compares AUC of scoring classifiers. Our policies output actions, not a score per class, so there is no ROC curve.

### 6.3 McNemar for lap success (slide 43)

For each track, each of the two policies either finished a lap or did not. That gives a 2 x 2 contingency table.

|  | B lapped | B failed |
|---|---|---|
| **A lapped** | both | A only (b) |
| **A failed** | B only (c) | neither |

Tracks where both succeed or both fail say nothing about which is better. Only the **discordant** tracks b and c matter. Under H0 a discordant track is equally likely to favour A or B, so b follows a Binomial(b + c, 0.5). Slide 43 gives the chi-square form M = (|b − c| − 1)² / (b + c), valid when b + c ≥ 20, and the binomial otherwise. We often have fewer than 20 discordant tracks, so `stats.py` always uses the **exact binomial** form, which is valid for any count.

Example. PPO seed 2 against pure pursuit at difficulty 0. Pure pursuit finished 8 tracks that PPO did not, PPO finished none that pure pursuit did not. b = 0, c = 8, p = 2 x 0.5⁸ = 0.0078. Pure pursuit's higher lap rate is significant.

McNemar compares two single models, so for PPO each training seed is tested separately against the baseline. When both sides have three seeds (full model against an ablation) seed i is paired with seed i, because they share the training seed.

### 6.4 Wilcoxon signed-rank for progress and lap time (slide 40)

For each track take the difference A − B. Rank the absolute differences, add the ranks of the positive ones and of the negative ones. If A and B were equivalent the two sums would be similar. It is non-parametric, uses the size of the differences and not only their sign, and is the test slide 40 recommends for skew, outliers and bounded metrics. For PPO with several seeds we first average each track over the seeds, so there is one paired value per track.

Example. PPO against pure pursuit at difficulty 0 on lap time, tracks both finished. Median difference −5.4 s, p < 0.0001. PPO is significantly faster when it finishes.

### 6.5 Many policies, Friedman and Holm (slide 42)

The final table compares PPO, its ablations and two baselines. Testing every pair at α = 0.05 inflates the type I error. With 4 models there are 6 pairs, and the chance of at least one false "significant" result is well above 5% (slide 42).

1. **Friedman test** on per-track progress across all policies. It ranks the policies on each track and asks whether some policy ranks consistently higher. If p ≥ α we stop, no difference shown.
2. If Friedman rejects, pairwise tests (McNemar and Wilcoxon above) tell **which** pairs differ. Slide 42 suggests a post hoc like Nemenyi. We use the **Holm** correction on all pairwise p-values instead. It controls the same family-wise error, is less conservative than Bonferroni, and needs only scipy. The "p Holm" column is the one to read for significance.

## 7. Limits we report openly

- **Three training seeds.** Enough to see that training is unstable, far too few to test differences between training methods at seed level. A Mann-Whitney test with 3 against 3 cannot reach p < 0.05 at all, its smallest possible p is 0.1. Our tests are per track, so they speak about the trained agents we have, and the seed t interval shows how wide the uncertainty about the method is.
- **Power.** With 50 tracks a lap rate interval is about ± 0.1 wide. The final test split uses 200 tracks to bring that to about ± 0.07. Small real differences between ablations may still not reach significance, which is a type II error risk (slide 38), not proof of equality.
- **Selection.** `best_model.zip` is chosen on validation tracks 1000 to 1019, so validation numbers are slightly optimistic. Only test numbers go in the final table.
- **Deterministic policy.** Each policy is run once per track with deterministic actions. The variation we measure comes from tracks and training seeds, not from action noise.

## 8. Current results with intervals and tests

Validation tracks 1000 to 1049, 3 PPO seeds of `mvp`. Full output in `runs/stats_val_d0.md` and `runs/stats_val_d1.md`.

| Difficulty | Policy | Lap rate | 95% CI over tracks | Per seed | Progress |
|---|---|---|---|---|---|
| 0 | PPO mvp | 0.61 | [0.58, 0.65] | 1.00, 0.00, 0.84 | 0.64 |
| 0 | pure pursuit | 1.00 | [0.93, 1.00] | - | 1.00 |
| 0 | random | 0.00 | [0.00, 0.07] | - | 0.00 |
| 1 | PPO mvp | 0.49 | [0.43, 0.56] | 0.78, 0.00, 0.70 | 0.59 |
| 1 | pure pursuit | 1.00 | [0.93, 1.00] | - | 1.00 |
| 1 | random | 0.00 | [0.00, 0.07] | - | 0.00 |

Friedman rejects at both difficulties (p < 0.0001), so the three policies differ. Pairwise, Holm corrected

- **PPO against random.** Seeds 0 and 2 lap significantly more tracks (McNemar, 50 and 42 discordant tracks at difficulty 0, p < 0.0001). Seed 1 is indistinguishable from random, 0 discordant tracks. Progress is higher (Wilcoxon p < 0.0001). Gate 1 holds for the method in two of three runs.
- **PPO against pure pursuit.** At difficulty 0 seed 0 ties with pure pursuit (no discordant track), seed 2 laps fewer tracks (p 0.023 after Holm). At difficulty 1 every PPO seed laps significantly fewer (p ≤ 0.002 after Holm). On tracks both finish, PPO is about 5 s faster (Wilcoxon p < 0.0001).
- **Reading.** PPO learns a faster but less reliable driver than the hand-written controller, and whether training works at all depends on the seed. Both statements are now backed by tests, not only by means.

## 9. Checklist for any reported number

- Same track seeds, same difficulty and same `--n` for every policy in a table
- Lap rate with a 95% interval and the three outcome rates, never lap rate alone
- PPO per seed next to the mean
- Lap time compared only on tracks both finished
- Paired tests, McNemar for laps and Wilcoxon for progress and time
- Friedman first and Holm corrected p-values when more than two policies are compared
- Effect size (discordant counts or median difference) next to every p-value
- Test split once, validation for everything else

## References

- Tema 8, Evaluación de Modelos. Selección de Métricas. J. L. Sancho, R. M. Menchón, UPCT 2026/2027
- Maier-Hein L. et al. Metrics reloaded. Nat Methods 21, 195 to 212 (2024)
- Reinke A. et al. Understanding metric-related pitfalls in image analysis validation. Nat Methods 21, 182 to 194 (2024)
- Rainio O., Teuho J., Klén R. Evaluation metrics and statistical tests for machine learning. Sci Rep 14, 6086 (2024)
