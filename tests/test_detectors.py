"""Detectors, health checks and the monitor on synthetic runs with a known truth."""

import numpy as np
import pytest

from evalfleet.detectors import (
    CORRECTED,
    DETECTORS,
    MonitorConfig,
    detection_delay,
    first_alert,
    grader_alerts,
    look_levels,
    regression_alerts,
    saturation_flags,
    twin_gap_alerts,
)
from evalfleet.monitor import load_csv, monitor_run, text_summary

CFG = MonitorConfig()


def stable_run(rng, n_ckpt=30, accuracies=(0.99, 0.9, 0.7, 0.4), n=200):
    """A run in which nothing changes: every item keeps its own chance of being right."""
    acc = np.asarray(accuracies)[:, None]
    p = rng.beta(acc * 0.5, (1 - acc) * 0.5, size=(len(accuracies), n))
    return rng.random((n_ckpt, len(accuracies), n)) < p


def frozen_run(n_ckpt=30, n_evals=4, n=200, right=160):
    """A run in which every checkpoint gives exactly the same answers."""
    x = np.zeros((n_ckpt, n_evals, n), dtype=bool)
    x[:, :, :right] = True
    return x


def test_look_levels_spend_exactly_alpha_after_the_first_look():
    levels = look_levels(30, CFG)
    assert (levels[: CFG.first_look] == 0).all()
    assert levels.sum() == pytest.approx(CFG.alpha)
    series = look_levels(30, MonitorConfig(spending="series"))
    assert series.sum() < CFG.alpha and series[CFG.first_look] == pytest.approx(CFG.alpha / 2)


def test_no_detector_alerts_on_a_frozen_run():
    alerts = regression_alerts(frozen_run(), CFG)
    assert set(alerts) == set(DETECTORS)
    assert not any(a.any() for a in alerts.values())


def test_nothing_is_flagged_before_the_first_look():
    x = frozen_run()
    x[2:4, 0, :120] = False  # a large dip, over before monitoring starts
    alerts = regression_alerts(x, CFG)
    assert not any(a.any() for a in alerts.values())


def test_a_planted_drop_is_caught_at_once_on_the_right_eval():
    x = frozen_run()
    x[15:, 1, :60] = False  # 30 points lost on eval 1 from checkpoint 15
    alerts = regression_alerts(x, CFG)
    for name in DETECTORS:
        hit = first_alert(alerts[name])
        expected = 16 if name == "mcnemar_confirm" else 15
        assert hit.tolist() == [-1, expected, -1, -1], name
    # The lag-4 comparison keeps the alert alive for four checkpoints, then it clears.
    assert alerts["mcnemar_seq"][15:19, 1].all() and not alerts["mcnemar_seq"][19:, 1].any()


def test_a_slow_decline_is_caught_by_the_longer_lag_only():
    x = frozen_run()
    for step in range(12):  # 5 more items lost at every checkpoint from 12 on
        x[12 + step :, 0, step * 5 : step * 5 + 5] = False
    alerts = regression_alerts(x, CFG)
    lag1_only = regression_alerts(x, MonitorConfig(lags=(1,)))
    assert alerts["mcnemar_seq"][:, 0].any()
    assert not lag1_only["mcnemar_seq"].any()


def test_pairing_sees_what_the_unpaired_test_misses():
    # 16 of 200 items lost and none gained: 0.5**16 is far below the corrected level, but
    # 100/200 against 84/200 is nothing special to an unpaired test.
    x = frozen_run(right=100)
    x[20:, 2, :16] = False
    alerts = regression_alerts(x, CFG)
    assert alerts["mcnemar_seq"][20, 2]
    assert not alerts["z_seq"].any()
    assert alerts["threshold"][20, 2]


def test_false_alarm_rates_on_a_synthetic_null():
    rng = np.random.default_rng(0)
    sims = 300
    alarmed = {d: 0 for d in DETECTORS}
    for _ in range(sims):
        for d, a in regression_alerts(stable_run(rng), CFG).items():
            alarmed[d] += int(a.any())
    for d in CORRECTED:
        assert alarmed[d] / sims <= 0.08, (d, alarmed[d])  # nominal 0.05 plus sampling slack
    # The naive detectors are far above it. The threshold rule is the worse of the two here:
    # items that keep their answers make the unpaired z-test conservative.
    assert alarmed["threshold"] / sims > 0.5
    assert alarmed["z_naive"] / sims > 0.1


def test_twin_check_flags_contamination_and_ignores_sampling_gaps():
    rng = np.random.default_rng(1)
    both = stable_run(rng, accuracies=(0.3, 0.3, 0.9, 0.9), n=400)
    main, twin = both[:, :, :200].copy(), both[:, :, 200:]
    assert not twin_gap_alerts(main, twin, CFG)[0].any()
    for t in range(14, 30):  # eval 0: four more main items memorised at every checkpoint
        main[t:, 0, : (t - 13) * 4] = True
    alerts, p_level, p_gain = twin_gap_alerts(main, twin, CFG)
    assert alerts[:, 0].any() and not alerts[:14].any() and not alerts[:, 1:].any()
    assert p_gain[29, 0] < 1e-4 and p_gain[29, 0] < p_level[29, 0] < 0.01


def test_grader_check_needs_more_than_the_tolerated_disagreement():
    score = frozen_run()
    regrade = score.copy()
    score[12:, 0, 0] = False  # one item in 200 disagrees: within tolerance
    assert not grader_alerts(score, regrade, CFG)[0].any()
    score[18:, 3, :20] = False  # twenty items disagree
    alerts, disagree = grader_alerts(score, regrade, CFG)
    assert first_alert(alerts).tolist() == [-1, -1, -1, 18]
    assert disagree[18, 3] == 20 and disagree[17, 3] == 0


def test_saturation_needs_three_checkpoints_at_the_ceiling():
    x = frozen_run(n_ckpt=6, n_evals=2, right=200)
    x[:, 1, :10] = False  # eval 1 sits at 95%
    x[2, 0, :10] = False  # eval 0 dips once
    flags = saturation_flags(x, CFG)
    assert flags[:, 0].tolist() == [False, False, False, False, False, True]
    assert not flags[:, 1].any()


def test_detection_delay_counts_from_the_first_affected_checkpoint():
    alerts = np.zeros((30, 3), dtype=bool)
    alerts[10, 1] = True  # before the onset: ignored
    alerts[17, 1] = True
    alerts[16, 2] = True  # another eval
    assert detection_delay(alerts, onset=14, evals=[1]) == (True, 3)
    assert detection_delay(alerts, onset=14, evals=[0, 1, 2]) == (True, 2)
    assert detection_delay(alerts, onset=14, evals=[0]) == (False, 16)
    assert detection_delay(alerts, onset=17, evals=[1]) == (False, 13)


def _verdict(report, name):
    return report.verdicts()[name]


def test_monitor_blames_the_model_for_a_real_regression():
    names = ("a", "b", "c", "d")
    main = frozen_run()
    twin = frozen_run()
    main[16:, 2, :50] = False
    twin[16:, 2, :46] = False
    report = monitor_run(main, np.arange(1, 31) * 80, names, twin=twin, regrade=main.copy())
    assert _verdict(report, "c") == ("model_regression",)
    assert all(_verdict(report, e) == ("ok",) for e in "abd")
    first = report.alerts[0]
    assert (first.kind, first.eval, first.step) == ("regression", "c", 17 * 80)
    assert "50 of 200 items went from right to wrong and 0 went from wrong to right" in first.text
    assert "fell 25.0 points between step 1280 and step 1360" in first.text
    assert "twin set fell too" in first.text
    assert "regression @ step 1360" in text_summary(report)


def test_monitor_blames_the_grader_when_the_regrade_is_unmoved():
    names = ("a", "b", "c", "d")
    regrade = frozen_run()
    score = regrade.copy()
    score[20:, 0, :30] = False  # the production grader starts marking 30 right answers wrong
    report = monitor_run(score, np.arange(30), names, twin=frozen_run(), regrade=regrade)
    assert _verdict(report, "a") == ("grader_fault",)
    kinds = {a.kind for a in report.alerts if a.eval == "a"}
    assert kinds == {"regression", "grader"}  # the score did fall; the verdict says why
    # Without the second grader the same data would be read as a model regression.
    blind = monitor_run(score, np.arange(30), names)
    assert _verdict(blind, "a") == ("model_regression",)


def test_monitor_flags_contamination_without_a_regression():
    names = ("a", "b", "c", "d")
    main = frozen_run(right=40)
    twin = frozen_run(right=40)
    for t in range(14, 30):
        main[t:, 3, 40 : 40 + (t - 13) * 5] = True
    report = monitor_run(main, np.arange(30), names, twin=twin, regrade=main.copy())
    assert _verdict(report, "d") == ("contamination",)
    assert [a.kind for a in report.alerts] == ["contamination"]
    assert "leaked into training" in report.alerts[0].text


def test_csv_loader_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    score = rng.random((5, 2, 7)) < 0.6
    path = tmp_path / "r.csv"
    lines = ["step,eval,item,correct"]
    for t in range(5):
        for k, name in enumerate(("alpha", "beta")):
            for i in range(7):
                lines.append(f"{(t + 1) * 100},{name},q{i},{int(score[t, k, i])}")
    path.write_text("\n".join(lines) + "\n")
    back, steps, evals = load_csv(path)
    assert (back == score).all() and steps.tolist() == [100, 200, 300, 400, 500] and evals == ("alpha", "beta")
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        load_csv(path)
