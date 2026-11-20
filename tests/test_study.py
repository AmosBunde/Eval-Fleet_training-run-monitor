"""The study pipeline on fabricated runs whose right answers are known."""

import json

import numpy as np
import pytest

from evalfleet import study
from evalfleet.evaluate import save_results
from evalfleet.faults import FaultSpec
from evalfleet.figures import make_figures
from evalfleet.skills import SKILL_NAMES
from evalfleet.train import RunSpec, TrainConfig

STEPS = TrainConfig().ckpt_steps
ONSET = 15  # index of the last clean checkpoint


def _base(rng):
    """A quiet run: six evals at 92% and two at 50%, items fixed, plus a little flicker."""
    x = np.zeros((30, 8, 2, 200), dtype=bool)
    x[:, :6, :, :184] = True
    x[:, 6:, :, :100] = True
    return x ^ (rng.random(x.shape) < 0.003)


def _write(tmp_path, run_id, fault, score, regrade=None):
    spec = RunSpec(run_id, 0, fault, TrainConfig())
    meta = {"spec": spec.to_dict(), "ckpt_steps": STEPS, "train_seconds": 1.0, "eval_seconds": 1.0}
    save_results(tmp_path / "runs" / f"{run_id}.npz", score, score if regrade is None else regrade, meta)


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("study")
    rng = np.random.default_rng(0)
    onset_step = STEPS[ONSET]
    for i in range(3):
        _write(tmp, f"clean-{i:02d}", FaultSpec(), _base(rng))
    for i in range(2):
        x = _base(rng)
        x[ONSET + 1 :, 1, :, :60] = False  # reverse collapses by 30 points at once
        _write(tmp, f"corrupt-{i:02d}", FaultSpec("corrupt", "reverse", onset_step), x)
        x = _base(rng)
        for t in range(ONSET + 1, 30):  # sort loses 3 more items per checkpoint
            x[t:, 2, :, (t - ONSET - 1) * 3 : (t - ONSET) * 3] = False
        _write(tmp, f"remove-{i:02d}", FaultSpec("remove", "sort", onset_step), x)
        x = _base(rng)
        x[ONSET + 1 : ONSET + 3, :, :, :40] = False  # every eval dips for two checkpoints
        _write(tmp, f"lr_spike-{i:02d}", FaultSpec("lr_spike", None, onset_step, lr_factor=10.0, lr_steps=40), x)
        x = _base(rng)
        for t in range(ONSET + 1, 30):  # main items of add are memorised; twin is untouched
            x[t:, 6, 0, 100 : 100 + (t - ONSET) * 2] = True
        _write(tmp, f"contaminate-{i:02d}", FaultSpec("contaminate", "add", onset_step), x)
        regrade = _base(rng)
        x = regrade.copy()
        x[ONSET + 1 :, 0, :, 100:140] = False  # the grader fails 40 right answers of copy
        _write(tmp, f"grader_bug-{i:02d}", FaultSpec("grader_bug", "copy", onset_step), x, regrade)
    old = study.SYNTHETIC_RUNS, study.SUBSET_REPEATS
    study.SYNTHETIC_RUNS, study.SUBSET_REPEATS = 30, 2
    try:
        out = study.summarise(tmp, log=lambda *_: None)
    finally:
        study.SYNTHETIC_RUNS, study.SUBSET_REPEATS = old
    return tmp, out


def test_summary_file_matches_what_was_returned(summary):
    tmp, out = summary
    assert json.loads((tmp / "summary.json").read_text()) == out
    assert out["runs"]["total"] == 13 and out["runs"]["by_kind"]["clean"] == 3
    assert out["config"]["evals"] == list(SKILL_NAMES)
    assert out["config"]["tests_per_run_paired"] == 22 * 8 * 2


def test_planted_faults_are_detected_with_the_right_delay(summary):
    det = summary[1]["detection"]
    corrupt = det["corrupt"]["detectors"]
    assert corrupt["mcnemar_seq"]["mean_delay_censored"] == 1.0
    assert corrupt["mcnemar_confirm"]["mean_delay_censored"] == 2.0
    assert corrupt["threshold"]["detected"] == 2
    assert det["corrupt"]["mean_true_effect"] == pytest.approx(0.30, abs=0.01)
    # The slow decline (1.5 points a checkpoint) needs the lag-4 comparison and takes longer.
    slow = det["remove"]["detectors"]
    assert slow["mcnemar_seq"]["detected"] == 2 and slow["mcnemar_seq"]["mean_delay_censored"] > 3
    assert slow["threshold"]["detected"] == 0
    assert det["lr_spike"]["detectors"]["mcnemar_seq"]["mean_delay_censored"] == 1.0
    assert det["grader_bug"]["detectors"]["grader"]["mean_delay_censored"] == 1.0
    assert det["contaminate"]["detectors"]["contamination"]["detected"] == 2
    assert det["contaminate"]["detectors"]["mcnemar_seq"]["detected"] == 0


def test_blame_goes_to_the_right_party(summary):
    att = summary[1]["attribution"]
    assert att["corrupt"]["model"] == 2 and att["remove"]["model"] == 2 and att["lr_spike"]["model"] == 2
    assert att["grader_bug"]["grader"] == 2
    assert att["contaminate"]["contamination"] == 2
    assert att["clean"]["none"] == 3
    assert att["measurement_faults_blamed_on_eval"]["count"] == 4
    assert att["real_regressions_blamed_on_eval"]["count"] == 0


def test_clean_runs_raise_nothing_and_every_hypothesis_is_judged(summary):
    out = summary[1]
    fa = out["false_alarms"]["real_clean"]["detectors"]
    assert all(v["run_level"]["count"] == 0 for v in fa.values())
    assert set(out["hypotheses"]) == {"H1", "H2a", "H2b", "H2c", "H3", "H4", "H5", "H6", "H7", "H8"}
    assert all(v["outcome"] in ("supported", "failed") for v in out["hypotheses"].values())
    assert out["hypotheses"]["H5"]["evidence"] == {
        "grader_bug_right": 2,
        "contaminate_right": 2,
        "real_regressions_blamed_on_eval": 0,
    }
    assert out["hypotheses"]["H1"]["outcome"] == "failed"  # these fabricated clean runs are too quiet


def test_effect_split_and_pooled_items(summary):
    out = summary[1]
    split = out["by_true_effect"]
    assert split["took_effect"]["n_runs"] + split["little_or_no_effect"]["n_runs"] == 6
    assert split["took_effect"]["detectors"]["mcnemar_seq"]["detected"] == split["took_effect"]["n_runs"]
    pooled = out["pooled_items"]
    assert pooled["n_items"] == 400 and pooled["detected"] == 6
    assert pooled["clean_runs_alarmed"]["count"] == 0
    assert pooled["min_detectable_drop"] == pytest.approx(13 / 400)


def test_planted_drops_have_a_floor_and_a_ceiling(summary):
    planted = summary[1]["planted_drops"]
    assert planted["sizes"]["0"]["n_cases"] == 3 * 6  # the six evals at 90% or more, three clean runs
    assert planted["sizes"]["0"]["caught_within_window"]["mcnemar_seq"] == 0.0
    assert planted["sizes"]["20"]["caught_at_once"]["mcnemar_seq"] == 1.0
    assert planted["sizes"]["5"]["caught_within_window"]["mcnemar_seq"] <= 0.2  # 10 items; 13 are needed
    # 16 items lost, almost none gained: enough for the paired test, not for the unpaired one.
    assert planted["sizes"]["8"]["caught_within_window"]["mcnemar_seq"] >= 0.9
    assert planted["sizes"]["8"]["caught_within_window"]["z_seq"] == 0.0
    assert planted["smallest_drop_caught_9_times_in_10"]["mcnemar_seq"] == 8


def test_category_names():
    assert study._category(set()) == "none"
    assert study._category({"model_regression"}) == "model"
    assert study._category({"grader_fault"}) == "grader"
    assert study._category({"contamination"}) == "contamination"
    assert study._category({"model_regression", "grader_fault"}) == "mixed"


def test_figures_are_written(summary, tmp_path):
    tmp, _ = summary
    # The trajectories figure names runs of the real matrix; the fabricated set has two per kind.
    import evalfleet.figures as figures

    figures.TRAJECTORY_RUNS = ["corrupt-01", "remove-01", "lr_spike-00", "contaminate-01", "grader_bug-00", "clean-00"]
    try:
        make_figures(tmp, tmp_path / "img", log=lambda *_: None)
    finally:
        figures.TRAJECTORY_RUNS = figures.DEFAULT_TRAJECTORY_RUNS
    names = sorted(p.name for p in (tmp_path / "img").glob("*.png"))
    assert names == [
        "attribution.png", "effect_vs_detection.png", "headline.png", "operating_curve.png",
        "planted_drops.png", "small_evals.png", "trajectories.png",
    ]
    assert all((tmp_path / "img" / n).stat().st_size > 10_000 for n in names)
