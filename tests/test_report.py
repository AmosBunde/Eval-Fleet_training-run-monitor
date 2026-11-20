"""The HTML report and the command line."""

import re

import numpy as np
import pytest

from evalfleet.cli import main
from evalfleet.evaluate import save_results
from evalfleet.faults import FaultSpec
from evalfleet.monitor import Alert, episodes, monitor_run
from evalfleet.report import render
from evalfleet.runner import result_path
from evalfleet.train import RunSpec, TrainConfig, validate_run_id


def _faulty_run():
    """30 checkpoints, 8 evals: a real regression on one eval and a grader fault on another."""
    regrade = np.zeros((30, 8, 2, 200), dtype=bool)
    regrade[:, :, :, :170] = True
    regrade[18:, 2, :, :40] = False  # the model forgets 40 items of eval 2
    score = regrade.copy()
    score[21:, 0, :, 100:150] = False  # the grader starts failing 50 right answers of eval 0
    return score, regrade


def test_episodes_group_consecutive_alerts():
    mk = lambda kind, name, i: Alert(kind, name, i, i * 80, "x: y")
    eps = episodes([mk("regression", "a", 10), mk("regression", "a", 11), mk("regression", "a", 12),
                    mk("regression", "a", 20), mk("grader", "a", 11), mk("regression", "b", 11)])
    got = [(a.kind, a.eval, a.index, repeats, last) for a, repeats, last in eps]
    assert got == [
        ("regression", "a", 10, 2, 960),
        ("grader", "a", 11, 0, 880),
        ("regression", "b", 11, 0, 880),
        ("regression", "a", 20, 0, 1600),
    ]


def test_report_is_self_contained_and_says_what_happened():
    score, regrade = _faulty_run()
    names = ("copy", "reverse", "sort", "depth", "max", "index", "add", "modadd")
    report = monitor_run(score[:, :, 0], np.arange(1, 31) * 80, names, twin=score[:, :, 1], regrade=regrade[:, :, 0])
    html = render(report, "demo <run>")
    assert html.startswith("<!doctype html>")
    assert "demo &lt;run&gt;" in html and "<run>" not in html
    # Nothing is fetched: no links, scripts, stylesheets or remote images.
    assert not re.search(r"https?://|<script|<link|<img|@import|url\(", html)
    assert html.count("<svg") >= 8
    assert "The model got worse on sort." in html
    assert "The eval itself looks faulty for copy" in html
    assert "40 of 200 items went from right to wrong and 0 went from wrong to right" in html
    assert "attributed to the grader" in html
    assert "Eval fault: grader" in html and "Model regression" in html


def test_cli_export_monitor_and_report(tmp_path, capsys):
    score, regrade = _faulty_run()
    spec = RunSpec("cli-demo", 1, FaultSpec(), TrainConfig())
    run = tmp_path / "cli-demo.npz"
    save_results(run, score, regrade, {"spec": spec.to_dict(), "ckpt_steps": spec.train.ckpt_steps})

    main(["monitor", "--run", str(run)])
    out = capsys.readouterr().out
    assert "Eval fault: grader" in out and "regression @ step 1520" in out

    main(["export", "--run", str(run), "--csv", str(tmp_path / "items.csv")])
    main(["monitor", "--csv", str(tmp_path / "items.csv")])
    out = capsys.readouterr().out
    # From the CSV alone there is no second grader, so both drops are read as regressions.
    assert out.count("Model regression") == 2 and "grader" not in out

    main(["report", "--run", str(run), "--out", str(tmp_path / "r.html")])
    assert "Run cli-demo" in (tmp_path / "r.html").read_text()


BAD_IDS = ["../escape", "a/b", "/abs", "..", ".", "", "a b", "x\\y", "run\n", "r;rm"]


def test_run_ids_that_could_leave_the_directory_are_refused(tmp_path):
    for good in ("clean-00", "lr_spike-01", "v1.2_demo"):
        assert validate_run_id(good) == good
    for bad in BAD_IDS:
        with pytest.raises(ValueError, match="unsafe run id"):
            validate_run_id(bad)
        with pytest.raises(ValueError):
            result_path(tmp_path, bad)


def test_report_refuses_a_result_file_with_a_crafted_run_id(tmp_path, monkeypatch, capsys):
    score, regrade = _faulty_run()
    spec = RunSpec("../../escaped", 1, FaultSpec(), TrainConfig())
    run = tmp_path / "inner" / "deep" / "crafted.npz"
    save_results(run, score, regrade, {"spec": spec.to_dict(), "ckpt_steps": spec.train.ckpt_steps})
    monkeypatch.chdir(run.parent)
    with pytest.raises(SystemExit, match="unsafe run id"):
        main(["report", "--run", str(run)])
    assert not (tmp_path / "escaped.html").exists()
    assert not list(tmp_path.rglob("*.html"))
    # An explicit output path is the caller's own choice and still works.
    main(["report", "--run", str(run), "--out", str(tmp_path / "ok.html")])
    assert (tmp_path / "ok.html").exists()
    with pytest.raises(SystemExit, match="unsafe run id"):
        main(["train", "--run-id", "../x", "--steps", "1"])
