import json

import numpy as np
import pytest

from evaluation.stats import (
    bootstrap_ci,
    compare,
    holm,
    load,
    mcnemar_exact,
    model_pairs,
    report,
    summarize,
    t_ci,
    track_matrix,
    wilcoxon_paired,
    wilson_ci,
)


def fake_eval(models, seeds=None):
    """eval.json dict from lists of (outcome, progress, lap_time) per model."""
    seeds = seeds or list(range(1000, 1000 + len(models[0])))
    episodes = [
        [
            {"seed": s, "outcome": o, "progress": p, "lap_time": t}
            for s, (o, p, t) in zip(seeds, m, strict=True)
        ]
        for m in models
    ]
    return {"seeds": seeds, "episodes": episodes}


LAP = ("lap", 1.0, 30.0)
STUCK = ("stuck", 0.1, None)


def test_wilson_known_value_and_edges():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0) and wilson_ci(10, 10)[0] < 1.0
    assert wilson_ci(0, 10)[0] == 0.0 and wilson_ci(0, 10)[1] > 0.0


def test_bootstrap_contains_mean_and_drops_nan():
    lo, hi = bootstrap_ci([0.0, 1.0, np.nan] * 20)
    assert lo < 0.5 < hi
    assert bootstrap_ci([0.7] * 10) == (pytest.approx(0.7), pytest.approx(0.7))


def test_t_ci_wide_with_three_seeds():
    lo, hi = t_ci([1.0, 0.0, 0.84])
    assert lo < 0 and hi > 1  # three seeds that disagree say little about the mean
    assert np.isnan(t_ci([0.5])[0])


def test_mcnemar_counts_only_discordant_tracks():
    a = [1] * 10 + [1] * 8 + [0] * 2
    b = [1] * 10 + [0] * 8 + [0] * 2
    m = mcnemar_exact(a, b)
    assert (m["a_only"], m["b_only"]) == (8, 0)
    assert m["p"] == pytest.approx(2 * 0.5**8)
    assert mcnemar_exact([1, 0], [1, 0])["p"] == 1.0


def test_wilcoxon_ties_and_nan():
    assert wilcoxon_paired([1, 2, 3], [1, 2, 3])["p"] == 1.0
    w = wilcoxon_paired(np.arange(20) + 1.0, np.zeros(20))
    assert w["p"] < 0.001 and w["n"] == 20
    assert wilcoxon_paired([1.0, np.nan], [0.0, 1.0])["n"] == 1


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_model_pairs():
    assert model_pairs(3, 3) == [(0, 0), (1, 1), (2, 2)]
    assert model_pairs(3, 1) == [(0, 0), (1, 0), (2, 0)]
    with pytest.raises(ValueError):
        model_pairs(3, 2)


def test_track_matrix_and_summary():
    d = fake_eval([[LAP, STUCK, LAP, LAP], [STUCK, STUCK, LAP, LAP]])
    assert track_matrix(d, "lap").tolist() == [[1, 0, 1, 1], [0, 0, 1, 1]]
    assert np.isnan(track_matrix(d, "lap_time")[0, 1])
    s = summarize(d)
    assert s["lap_rate"] == pytest.approx(5 / 8) and s["lap_rate_per_seed"] == [0.75, 0.5]
    single = summarize(fake_eval([[LAP, STUCK]]))
    assert "lap_rate_seed_ci" not in single and single["lap_rate_ci"][0] > 0


def test_compare_rejects_unpaired_tracks():
    a = fake_eval([[LAP, LAP]], seeds=[1000, 1001])
    b = fake_eval([[LAP, LAP]], seeds=[1000, 1002])
    with pytest.raises(ValueError):
        compare(a, b)


def test_compare_and_report(tmp_path):
    good = fake_eval([[LAP] * 12, [LAP] * 11 + [STUCK]])
    bad = fake_eval([[STUCK] * 12])
    c = compare(good, bad)
    assert len(c["mcnemar"]) == 2 and c["mcnemar"][0]["a_only"] == 12
    assert c["lap_time_wilcoxon"]["n"] == 0  # bad never lapped, nothing to pair
    third = fake_eval([[LAP] * 6 + [STUCK] * 6])
    text = report({"good": good, "bad": bad, "third": third})
    assert "Friedman" in text and "p Holm" in text
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"metrics": {}}))
    with pytest.raises(ValueError):
        load(path)
