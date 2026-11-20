"""The statistics against values worked out by hand or taken from standard examples."""

import numpy as np
import pytest

from evalfleet import stats


def test_wilson_interval_known_value():
    # 8 successes in 10 trials at 95%: the standard Wilson interval is (0.4902, 0.9433).
    lo, hi = stats.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_wilson_interval_never_leaves_unit_range():
    lo, hi = stats.wilson_interval(np.array([0, 200]), 200)
    assert lo[0] == pytest.approx(0, abs=1e-12) and hi[1] == pytest.approx(1, abs=1e-12)
    assert 0 < hi[0] < 0.03 and 0.97 < lo[1] < 1


def test_two_proportion_z_known_value():
    # 60/100 against 45/100: pooled p = 0.525, se = 0.070622, z = 2.1240, one-sided p = 0.01684.
    p = stats.two_proportion_z_pvalue(60, 100, 45, 100)
    assert float(p) == pytest.approx(0.01684, abs=2e-5)
    assert float(stats.two_proportion_z_pvalue(50, 100, 50, 100)) == pytest.approx(0.5)


def test_two_proportion_z_handles_zero_variance():
    assert float(stats.two_proportion_z_pvalue(100, 100, 100, 100)) == pytest.approx(0.5)


def test_mcnemar_exact_known_value():
    # 10 lost, 2 gained: P(X >= 10 | n=12, p=1/2) = (66 + 12 + 1) / 4096.
    assert float(stats.mcnemar_exact_pvalue(10, 2)) == pytest.approx(79 / 4096)
    # No discordant items: no evidence at all.
    assert float(stats.mcnemar_exact_pvalue(0, 0)) == pytest.approx(1.0)
    # 14 lost, 0 gained: 0.5 ** 14.
    assert float(stats.mcnemar_exact_pvalue(14, 0)) == pytest.approx(0.5**14)


def test_discordant_counts():
    ref = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    new = np.array([1, 0, 0, 1, 0, 1], dtype=bool)
    lost, gained = stats.discordant_counts(ref, new)
    assert (int(lost), int(gained)) == (2, 1)


def test_fisher_exact_tea_tasting():
    # Fisher's tea-tasting table (3 of 4 against 1 of 4): one-sided p = 17/70.
    assert float(stats.fisher_exact_greater(3, 4, 1, 4)) == pytest.approx(17 / 70)
    assert float(stats.fisher_exact_greater(4, 4, 0, 4)) == pytest.approx(1 / 70)


def test_binomial_tail_known_value():
    # P(X >= 2) for Binomial(10, 0.1) = 1 - 0.9**10 - 10 * 0.1 * 0.9**9.
    expected = 1 - 0.9**10 - 10 * 0.1 * 0.9**9
    assert float(stats.binomial_tail_pvalue(2, 10, 0.1)) == pytest.approx(expected)
    assert float(stats.binomial_tail_pvalue(0, 10, 0.1)) == pytest.approx(1.0)


def test_betting_evalue_hand_computed():
    # One bet size 0.5, no margin, 3 lost, 1 gained, 6 unchanged: 1.5**3 * 0.5 = 1.6875.
    e = stats.betting_evalue(3, 1, 10, margin=0.0, lambdas=[0.5])
    assert float(e) == pytest.approx(1.6875)
    # No discordant items and no margin: the bet neither wins nor loses.
    assert float(stats.betting_evalue(0, 0, 50, margin=0.0)) == pytest.approx(1.0)
    # With a margin, unchanged items count as (weak) evidence against a drop.
    assert float(stats.betting_evalue(0, 0, 50, margin=0.02)) < 1.0


def test_betting_evalue_is_valid_at_the_null_boundary():
    # Items drop by exactly the margin on average. E[e] must be <= 1 and P(e >= 20) <= 0.05.
    rng = np.random.default_rng(0)
    n, margin, sims = 200, 0.02, 40000
    u = rng.random((sims, n))
    lost = (u < 0.06).sum(1)
    gained = ((u >= 0.06) & (u < 0.10)).sum(1)
    e = stats.betting_evalue(lost, gained, n, margin)
    # (The sample mean sits below 1 because the distribution of e is heavy-tailed.)
    assert e.mean() <= 1.05
    assert (e >= 20).mean() <= 0.05


def test_mcnemar_holds_its_level_under_the_null():
    rng = np.random.default_rng(1)
    u = rng.random((40000, 200))
    lost = (u < 0.05).sum(1)
    gained = ((u >= 0.05) & (u < 0.10)).sum(1)
    assert (stats.mcnemar_exact_pvalue(lost, gained) < 0.05).mean() <= 0.05


def test_benjamini_hochberg_textbook_example():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert stats.benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8


def test_benjamini_hochberg_is_step_up():
    # The largest p-value passes its threshold, so every hypothesis is rejected even though
    # the smallest one fails its own.
    p = [0.05, 0.02, 0.045, 0.03, 0.04]
    assert stats.benjamini_hochberg(p, 0.05).all()
    assert not stats.benjamini_hochberg([0.2, 0.5], 0.05).any()


def test_spending_schedules_never_exceed_alpha():
    for kind in ("uniform", "series"):
        for looks in (1, 7, 500):
            sched = stats.spending_schedule(0.05, looks, kind)
            assert sched.sum() <= 0.05 + 1e-12
            assert (sched > 0).all()
    assert stats.spending_schedule(0.05, 4, "series").tolist() == pytest.approx(
        [0.05 / 2, 0.05 / 6, 0.05 / 12, 0.05 / 20]
    )
    assert stats.bonferroni_alpha(0.05, 10) == pytest.approx(0.005)


def test_split_half_reliability_extremes():
    rng = np.random.default_rng(2)
    # Every item is right with the same probability, which climbs across checkpoints.
    climbing = rng.random((30, 400)) < np.linspace(0.1, 0.9, 30)[:, None]
    assert stats.split_half_reliability(climbing) > 0.95
    # No signal: the same probability at every checkpoint.
    flat = rng.random((30, 400)) < 0.5
    assert abs(stats.split_half_reliability(flat)) < 0.5
    # Nothing varies at all: undefined.
    assert np.isnan(stats.split_half_reliability(np.ones((10, 20), dtype=bool)))


def test_bootstrap_ci_covers_the_mean():
    lo, hi = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=4000)
    assert lo < 3.0 < hi
    assert 1.0 <= lo and hi <= 5.0


def test_minimum_detectable_drop():
    # At level 1e-3 the sign test needs 10 losses in a row (0.5**10 < 1e-3): 10 of 200 items.
    assert stats.minimum_detectable_drop(200, 1e-3) == pytest.approx(0.05)
