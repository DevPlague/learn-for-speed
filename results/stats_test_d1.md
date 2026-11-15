| policy | models | tracks | lap rate | 95% CI | progress | 95% CI |
|---|---|---|---|---|---|---|
| final | 3 | 200 | 0.96 | [0.94, 0.97] | 0.98 | [0.97, 0.99] |
|  | per seed 0.96, 0.97, 0.94 | | | seed t [0.91, 1.00] | | |
| pure_pursuit | 1 | 200 | 1.00 | [0.98, 1.00] | 1.00 | [1.00, 1.00] |
| random | 1 | 200 | 0.00 | [0.00, 0.02] | 0.00 | [0.00, 0.00] |

| test | detail | p | p Holm | significant |
|---|---|---|---|---|
| final vs pure_pursuit lap McNemar models (0, 0) | 0 vs 7 discordant | 0.0156 | 0.0312 | yes |
| final vs pure_pursuit lap McNemar models (1, 0) | 0 vs 6 discordant | 0.0312 | 0.0312 | yes |
| final vs pure_pursuit lap McNemar models (2, 0) | 0 vs 13 discordant | 0.0002 | 0.0007 | yes |
| final vs pure_pursuit progress Wilcoxon | n 200, median diff +0.00 | 0.0000 | 0.0001 | yes |
| final vs pure_pursuit lap time both lapped Wilcoxon | n 200, median diff -4.42 | 0.0000 | 0.0000 | yes |
| final vs random lap McNemar models (0, 0) | 193 vs 0 discordant | 0.0000 | 0.0000 | yes |
| final vs random lap McNemar models (1, 0) | 194 vs 0 discordant | 0.0000 | 0.0000 | yes |
| final vs random lap McNemar models (2, 0) | 187 vs 0 discordant | 0.0000 | 0.0000 | yes |
| final vs random progress Wilcoxon | n 200, median diff +1.00 | 0.0000 | 0.0000 | yes |
| final vs random lap time both lapped Wilcoxon | n 0, median diff +0.00 | n/a | n/a | - |
| pure_pursuit vs random lap McNemar models (0, 0) | 200 vs 0 discordant | 0.0000 | 0.0000 | yes |
| pure_pursuit vs random progress Wilcoxon | n 200, median diff +1.00 | 0.0000 | 0.0000 | yes |
| pure_pursuit vs random lap time both lapped Wilcoxon | n 0, median diff +0.00 | n/a | n/a | - |

Friedman on per-track progress over 3 policies, p 0.0000
