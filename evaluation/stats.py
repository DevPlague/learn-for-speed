"""Confidence intervals and statistical tests between policies, from eval.json files.

Every policy is evaluated on the same track seeds, so the data are paired by track. The track
is the sample unit. Theory and choices are explained in docs/EVALUATION_STATISTICS.md.

Example
    make run CMD="python -m evaluation.stats ppo=runs/eval_ppo_val_d0/eval.json pure_pursuit=runs/eval_pure_pursuit_val_d0/eval.json"
"""

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ALPHA = 0.05


def load(path):
    """eval.json as a dict. Needs the per-track `episodes` written since the stats update."""
    data = json.loads(Path(path).read_text())
    if "episodes" not in data:
        raise ValueError(f"{path} has no per-track episodes, rerun evaluate")
    return data


def track_matrix(data, key):
    """Array models x tracks. `lap` gives 1 or 0, `progress` and `lap_time` give floats or NaN."""
    rows = []
    for model in data["episodes"]:
        if key == "lap":
            rows.append([float(e["outcome"] == "lap") for e in model])
        else:
            rows.append([np.nan if e[key] is None else e[key] for e in model])
    return np.array(rows, dtype=float)


def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for a proportion k of n. Stays inside [0, 1], fine at 0% and 100%."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_ci(values, confidence=0.95, n_boot=10_000, seed=0):
    """Percentile bootstrap interval of the mean, resampling tracks. NaN values are dropped."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values))).mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    return (float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail)))


def t_ci(values, confidence=0.95):
    """Student t interval of the mean over training seeds. Wide with 3 seeds, which is honest."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (np.nan, np.nan)
    half = stats.t.ppf(1 - (1 - confidence) / 2, len(values) - 1) * stats.sem(values)
    return (float(values.mean() - half), float(values.mean() + half))


def summarize(data):
    """Lap rate and progress with 95% intervals. One model uses Wilson, several add a seed t interval."""
    laps = track_matrix(data, "lap")
    progress = track_matrix(data, "progress")
    n_models, n_tracks = laps.shape
    out = {"n_models": n_models, "n_tracks": n_tracks, "lap_rate": float(laps.mean())}
    if n_models == 1:
        out["lap_rate_ci"] = wilson_ci(int(laps.sum()), n_tracks)
    else:
        out["lap_rate_ci"] = bootstrap_ci(laps.mean(axis=0))
        lo, hi = t_ci(laps.mean(axis=1))
        out["lap_rate_seed_ci"] = (max(0.0, lo), min(1.0, hi))  # clipped to the rate range
        out["lap_rate_per_seed"] = [float(v) for v in laps.mean(axis=1)]
    out["progress"] = float(progress.mean())
    out["progress_ci"] = bootstrap_ci(progress.mean(axis=0))
    return out


def mcnemar_exact(a, b):
    """Exact McNemar on two paired 0/1 vectors. Only discordant tracks count.

    b10 tracks where A lapped and B did not, b01 the reverse. Under H0 both are equally likely,
    so b10 follows Binomial(b10 + b01, 0.5). Exact because we often have fewer than 20 discordant.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    b10, b01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = 1.0 if b10 + b01 == 0 else float(stats.binomtest(b10, b10 + b01, 0.5).pvalue)
    return {"a_only": b10, "b_only": b01, "p": p}


def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank on paired values, NaN pairs dropped. p is 1 when all pairs tie."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    diff = a - b
    if len(diff) == 0 or np.all(diff == 0):
        return {"n": len(diff), "median_diff": 0.0, "p": 1.0}
    p = float(stats.wilcoxon(a, b).pvalue)
    return {"n": len(diff), "median_diff": float(np.median(diff)), "p": p}


def model_pairs(n_a, n_b):
    """Which models of A and B to pair for McNemar. Same count pairs by seed index, one side broadcasts."""
    if n_a == n_b:
        return [(i, i) for i in range(n_a)]
    if n_a == 1 or n_b == 1:
        return [(min(i, n_a - 1), min(i, n_b - 1)) for i in range(max(n_a, n_b))]
    raise ValueError("cannot pair models, give the same number of seeds or one model on a side")


def compare(a, b):
    """Paired comparison of two evals on the same tracks."""
    if [e["seed"] for e in a["episodes"][0]] != [e["seed"] for e in b["episodes"][0]]:
        raise ValueError("evals ran on different tracks, they are not paired")
    laps_a, laps_b = track_matrix(a, "lap"), track_matrix(b, "lap")
    pairs = model_pairs(len(laps_a), len(laps_b))
    mcnemar = [{"models": (i, j), **mcnemar_exact(laps_a[i], laps_b[j])} for i, j in pairs]
    with warnings.catch_warnings():  # a track no model lapped gives NaN, which is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        time_a = np.nanmean(track_matrix(a, "lap_time"), axis=0)
        time_b = np.nanmean(track_matrix(b, "lap_time"), axis=0)
    return {
        "mcnemar": mcnemar,
        "lap_wilcoxon": wilcoxon_paired(laps_a.mean(axis=0), laps_b.mean(axis=0)),
        "progress_wilcoxon": wilcoxon_paired(
            track_matrix(a, "progress").mean(axis=0), track_matrix(b, "progress").mean(axis=0)
        ),
        "lap_time_wilcoxon": wilcoxon_paired(time_a, time_b),
    }


def friedman(evals):
    """Friedman test on per-track mean progress of three or more policies."""
    columns = [track_matrix(d, "progress").mean(axis=0) for d in evals.values()]
    return float(stats.friedmanchisquare(*columns).pvalue)


def holm(pvalues):
    """Holm step-down correction. Same order as the input, values capped at 1."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return [float(v) for v in adjusted]


def fmt_ci(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def report(evals):
    """Markdown summary with intervals, pairwise tests with Holm correction and Friedman."""
    lines = [
        "| policy | models | tracks | lap rate | 95% CI | progress | 95% CI |",
        "|---" * 7 + "|",
    ]
    for name, data in evals.items():
        s = summarize(data)
        lines.append(
            f"| {name} | {s['n_models']} | {s['n_tracks']} | {s['lap_rate']:.2f} | "
            f"{fmt_ci(s['lap_rate_ci'])} | {s['progress']:.2f} | {fmt_ci(s['progress_ci'])} |"
        )
        if "lap_rate_seed_ci" in s:
            per_seed = ", ".join(f"{v:.2f}" for v in s["lap_rate_per_seed"])
            lines.append(
                f"|  | per seed {per_seed} | | | seed t {fmt_ci(s['lap_rate_seed_ci'])} | | |"
            )

    tests = []
    for (na, a), (nb, b) in itertools.combinations(evals.items(), 2):
        c = compare(a, b)
        for m in c["mcnemar"]:
            tests.append(
                (
                    f"{na} vs {nb} lap McNemar models {m['models']}",
                    m["p"],
                    f"{m['a_only']} vs {m['b_only']} discordant",
                )
            )
        for key, label in (
            ("progress_wilcoxon", "progress"),
            ("lap_time_wilcoxon", "lap time both lapped"),
        ):
            w = c[key]
            detail = f"n {w['n']}, median diff {w['median_diff']:+.2f}"
            tests.append((f"{na} vs {nb} {label} Wilcoxon", w["p"] if w["n"] else None, detail))
    run = [p for _, p, _ in tests if p is not None]  # tests with no pairs are not counted in Holm
    adjusted = iter(holm(run)) if run else iter([])
    lines += ["", "| test | detail | p | p Holm | significant |", "|---|---|---|---|---|"]
    for label, p, detail in tests:
        if p is None:
            lines.append(f"| {label} | {detail} | n/a | n/a | - |")
            continue
        p_adj = next(adjusted)
        sig = "yes" if p_adj < ALPHA else "no"
        lines.append(f"| {label} | {detail} | {p:.4f} | {p_adj:.4f} | {sig} |")
    if len(evals) >= 3:
        lines += [
            "",
            f"Friedman on per-track progress over {len(evals)} policies, p {friedman(evals):.4f}",
        ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evals", nargs="+", help="name=path/to/eval.json, all on the same tracks")
    p.add_argument("--out", default=None, help="optional Markdown file for the report")
    args = p.parse_args()
    evals = {k: load(v) for k, v in (item.split("=", 1) for item in args.evals)}
    text = report(evals)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
