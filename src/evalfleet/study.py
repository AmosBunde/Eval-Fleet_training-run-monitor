"""The study: score every detector against the known faults and write results/summary.json.

Every number quoted in README.md and REPORT.md is read from the file this module writes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import stats
from .detectors import (
    CORRECTED,
    DETECTORS,
    MonitorConfig,
    detection_delay,
    grader_alerts,
    pair_tables,
    regression_alerts,
    regression_pvalues,
    saturation_flags,
    twin_gap_alerts,
)
from .evaluate import RunResult, load_results
from .faults import MEASUREMENT_FAULTS, REAL_REGRESSIONS
from .monitor import monitor_result

SUBSET_SIZES = (25, 50, 100, 200)
SUBSET_REPEATS = 20
SYNTHETIC_RUNS = 2000
SYNTHETIC_ACCURACY = (0.995, 0.98, 0.95, 0.9, 0.75, 0.5, 0.2, 0.06)
EFFECT_FLOOR = 0.05  # a fault "took effect" if the twin-set score really fell this much


def load_all(results_dir: Path) -> list[RunResult]:
    return [load_results(p) for p in sorted((results_dir / "runs").glob("*.npz"))]


def _round(x, digits: int = 4):
    if isinstance(x, dict):
        return {k: _round(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, digits) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), digits)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def affected_evals(run: RunResult) -> list[int]:
    if run.fault.kind == "clean":
        return []
    if run.fault.skill is None:
        return list(range(len(run.skills)))
    return [run.skills.index(run.fault.skill)]


def _rate(k: int, n: int) -> dict:
    lo, hi = stats.wilson_interval(k, n) if n else (np.nan, np.nan)
    return {"count": int(k), "n": int(n), "rate": k / n if n else np.nan, "ci95": [float(lo), float(hi)]}


# --- false alarms ---------------------------------------------------------------------------


def clean_false_alarms(runs: list[RunResult], cfg: MonitorConfig, item_set: int = 0) -> dict:
    """Run-level and cell-level false alerts of every detector on the clean runs."""
    per_det = {d: {"runs_alarmed": 0, "cells": 0, "episodes": 0} for d in DETECTORS}
    health = {"contamination": 0, "grader": 0}
    n_cells = 0
    for run in runs:
        det = regression_alerts(run.score[:, :, item_set], cfg)
        n_cells = (run.score.shape[0] - cfg.first_look) * run.score.shape[1]
        for d in DETECTORS:
            per_det[d]["runs_alarmed"] += int(det[d].any())
            per_det[d]["cells"] += int(det[d].sum())
            per_det[d]["episodes"] += int(det[d].any(0).sum())
        health["contamination"] += int(twin_gap_alerts(run.score[:, :, item_set], run.score[:, :, 1 - item_set], cfg)[0].any())
        health["grader"] += int(grader_alerts(run.score[:, :, item_set], run.regrade[:, :, item_set], cfg)[0].any())
    n = len(runs)
    out = {"n_runs": n, "looks_per_run": n_cells // max(1, len(runs[0].skills)) if runs else 0, "detectors": {}}
    for d in DETECTORS:
        r = per_det[d]
        out["detectors"][d] = {
            "run_level": _rate(r["runs_alarmed"], n),
            "false_alerts_per_run": r["cells"] / n,
            "evals_alarmed_per_run": r["episodes"] / n,
            "cell_rate": r["cells"] / (n * n_cells),
        }
    out["health"] = {k: _rate(v, n) for k, v in health.items()}
    return out


def synthetic_null(cfg: MonitorConfig, n_runs: int = SYNTHETIC_RUNS, seed: int = 7) -> dict:
    """False-alarm rates when nothing changes at all.

    Each simulated run has 30 checkpoints, 8 evals and 200 main plus 200 twin items. An
    item has its own fixed probability of being answered correctly (drawn from a U-shaped
    Beta, so most items are reliably right or reliably wrong) and is redrawn independently
    at every checkpoint. The second grader disagrees with the first at exactly the
    tolerated rate. Every null hypothesis the monitor tests is therefore true.
    """
    rng = np.random.default_rng(seed)
    n_ckpt, n = 30, 200
    acc = np.asarray(SYNTHETIC_ACCURACY)[:, None]
    counts = Counter()
    for _ in range(n_runs):
        p = rng.beta(acc * 0.5, (1 - acc) * 0.5, size=(len(acc), 2 * n))
        x = rng.random((n_ckpt, len(acc), 2 * n)) < p
        main, twin = x[:, :, :n], x[:, :, n:]
        regrade = main ^ (rng.random(main.shape) < cfg.grader_tolerance)
        for d, a in regression_alerts(main, cfg).items():
            counts[d] += int(a.any())
        counts["contamination"] += int(twin_gap_alerts(main, twin, cfg)[0].any())
        counts["grader"] += int(grader_alerts(main, regrade, cfg)[0].any())
    return {
        "n_runs": n_runs,
        "eval_accuracies": list(SYNTHETIC_ACCURACY),
        "run_level": {k: _rate(counts[k], n_runs) for k in list(DETECTORS) + ["contamination", "grader"]},
    }


# --- detection ------------------------------------------------------------------------------


def true_effect(run: RunResult) -> float:
    """Size of the real change, measured on the twin set with the reference grader.

    For a fault on one skill: twin score at the onset minus the mean twin score afterwards
    (positive means the model got worse). For a learning-rate spike: the largest drop from
    the onset to any later checkpoint, over all evals.
    """
    o = run.onset_index
    twin = run.regrade[:, :, 1].mean(-1)
    if run.fault.skill is None:
        return float((twin[o] - twin[o + 1 :].min(0)).max())
    k = run.skills.index(run.fault.skill)
    return float(twin[o, k] - twin[o + 1 :, k].mean())


def per_run_table(runs: list[RunResult], cfg: MonitorConfig) -> list[dict]:
    rows = []
    for run in runs:
        det = regression_alerts(run.score[:, :, 0], cfg)
        report = monitor_result(run, cfg)
        verdicts = report.verdicts()
        row = {
            "run_id": run.run_id,
            "kind": run.fault.kind,
            "skill": run.fault.skill,
            "onset_step": run.fault.onset_step,
            "onset_index": run.onset_index,
            "train_seconds": run.meta.get("train_seconds"),
            "eval_seconds": run.meta.get("eval_seconds"),
            "verdicts": {k: list(v) for k, v in verdicts.items() if v != ("ok",)},
        }
        if run.fault.kind != "clean":
            o, evals = run.onset_index, affected_evals(run)
            row["true_effect"] = true_effect(run)
            row["checkpoints_after_onset"] = len(run.steps) - 1 - o
            row["delay"] = {}
            row["detected"] = {}
            for d in DETECTORS:
                hit, delay = detection_delay(det[d], o, evals)
                row["detected"][d], row["delay"][d] = hit, delay
            c_alert = twin_gap_alerts(run.score[:, :, 0], run.score[:, :, 1], cfg)[0]
            g_alert = grader_alerts(run.score[:, :, 0], run.regrade[:, :, 0], cfg)[0]
            for name, a in (("contamination", c_alert), ("grader", g_alert)):
                hit, delay = detection_delay(a, o, evals)
                row["detected"][name], row["delay"][name] = hit, delay
            row["attribution"] = attribution(run, verdicts)
        else:
            labels = {v for vs in verdicts.values() for v in vs} - {"ok"}
            row["attribution"] = _category(labels)
        rows.append(row)
    return rows


def _category(labels: set[str]) -> str:
    if not labels:
        return "none"
    if labels == {"model_regression"}:
        return "model"
    if labels == {"grader_fault"}:
        return "grader"
    if labels == {"contamination"}:
        return "contamination"
    return "mixed"


def attribution(run: RunResult, verdicts: dict[str, tuple[str, ...]]) -> str:
    """Which party the monitor blamed for the affected eval(s) of a fault run."""
    labels: set[str] = set()
    for k in affected_evals(run):
        labels |= set(verdicts[run.skills[k]])
    return _category(labels - {"ok"})


def detection_summary(rows: list[dict]) -> dict:
    out: dict = {}
    for kind in REAL_REGRESSIONS + MEASUREMENT_FAULTS:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        block = {
            "n_runs": len(sel),
            "mean_true_effect": float(np.mean([r["true_effect"] for r in sel])),
            "detectors": {},
        }
        for d in list(DETECTORS) + ["contamination", "grader"]:
            delays = [r["delay"][d] for r in sel]
            hits = [r["detected"][d] for r in sel]
            found = [dl for dl, h in zip(delays, hits) if h]
            block["detectors"][d] = {
                "detected": int(sum(hits)),
                "power": float(np.mean(hits)),
                "mean_delay_censored": float(np.mean(delays)),
                "mean_delay_ci95": list(stats.bootstrap_ci(delays, seed=1)),
                "median_delay_when_detected": float(np.median(found)) if found else np.nan,
                "detected_at_first_checkpoint": int(sum(dl == 1 and h for dl, h in zip(delays, hits))),
            }
        out[kind] = block
    return out


def paired_vs_unpaired(rows: list[dict], runs: list[RunResult], cfg: MonitorConfig) -> dict:
    """Delay of the unpaired z-test minus delay of the paired exact test, per regression run."""
    sel = [r for r in rows if r["kind"] in REAL_REGRESSIONS]
    diff = np.array([r["delay"]["z_seq"] - r["delay"]["mcnemar_seq"] for r in sel], dtype=float)
    out = {
        "n_runs": len(sel),
        "mean_delay_z_seq": float(np.mean([r["delay"]["z_seq"] for r in sel])),
        "mean_delay_mcnemar_seq": float(np.mean([r["delay"]["mcnemar_seq"] for r in sel])),
        "mean_difference": float(diff.mean()),
        "difference_ci95": list(stats.bootstrap_ci(diff, seed=2)),
        "paired_earlier": int((diff > 0).sum()),
        "same": int((diff == 0).sum()),
        "paired_later": int((diff < 0).sum()),
    }
    # The same comparison restricted to runs where the fault did not flatten the skill at once.
    gradual = [r for r in sel if r["kind"] != "corrupt"]
    gdiff = np.array([r["delay"]["z_seq"] - r["delay"]["mcnemar_seq"] for r in gradual], dtype=float)
    out["without_corrupt"] = {
        "n_runs": len(gradual),
        "mean_difference": float(gdiff.mean()) if gdiff.size else np.nan,
        "difference_ci95": list(stats.bootstrap_ci(gdiff, seed=3)),
        "paired_earlier": int((gdiff > 0).sum()),
        "same": int((gdiff == 0).sum()),
        "paired_later": int((gdiff < 0).sum()),
    }
    out["operating_curve"] = operating_curve(runs, cfg)
    return out


def operating_curve(runs: list[RunResult], cfg: MonitorConfig) -> dict:
    """Sweep the per-test threshold of the paired and unpaired tests.

    For each threshold: the share of clean runs with any false alert, and the mean
    (censored) detection delay and power over the real-regression runs. This compares the
    two tests at equal measured false-alarm rates instead of equal nominal levels.
    """
    thresholds = np.logspace(-8, np.log10(0.05), 36)
    clean = [r for r in runs if r.fault.kind == "clean"]
    faulty = [r for r in runs if r.fault.kind in REAL_REGRESSIONS]
    pvals = {r.run_id: regression_pvalues(pair_tables(r.score[:, :, 0], cfg)) for r in clean + faulty}
    curve: dict = {"thresholds": thresholds.tolist()}
    for test in ("z", "mcnemar"):
        fa, delay, power = [], [], []
        for tau in thresholds:
            fa.append(float(np.mean([(pvals[r.run_id][test] < tau).any() for r in clean])))
            ds, hs = [], []
            for r in faulty:
                hit, d = detection_delay((pvals[r.run_id][test] < tau).any(0), r.onset_index, affected_evals(r))
                ds.append(d)
                hs.append(hit)
            delay.append(float(np.mean(ds)))
            power.append(float(np.mean(hs)))
        curve[test] = {"clean_run_false_alarm": fa, "mean_delay": delay, "power": power}
    # Delay at matched measured false-alarm rates: the loosest threshold not exceeding each target.
    matched = {}
    for target in (0.125, 0.25, 0.5):
        entry = {}
        for test in ("z", "mcnemar"):
            ok = [i for i, f in enumerate(curve[test]["clean_run_false_alarm"]) if f <= target + 1e-9]
            i = max(ok) if ok else 0
            entry[test] = {
                "threshold": float(thresholds[i]),
                "clean_run_false_alarm": curve[test]["clean_run_false_alarm"][i],
                "mean_delay": curve[test]["mean_delay"][i],
                "power": curve[test]["power"][i],
            }
        matched[f"{target:g}"] = entry
    curve["matched"] = matched
    return curve


def by_true_effect(rows: list[dict]) -> dict:
    """Detection by the primary detector, split by whether the fault really moved the model.

    Added after the runs (see Amendments in PLAN.md): several injected faults turned out
    to change the twin-set score by less than ``EFFECT_FLOOR``, so there was little or
    nothing to detect.
    """
    out: dict = {"effect_floor": EFFECT_FLOOR}
    reg = [r for r in rows if r["kind"] in REAL_REGRESSIONS]
    for name, sel in (
        ("took_effect", [r for r in reg if r["true_effect"] >= EFFECT_FLOOR]),
        ("little_or_no_effect", [r for r in reg if r["true_effect"] < EFFECT_FLOOR]),
    ):
        block = {
            "n_runs": len(sel),
            "by_kind": dict(Counter(r["kind"] for r in sel)),
            "mean_true_effect": float(np.mean([r["true_effect"] for r in sel])) if sel else np.nan,
            "detectors": {},
        }
        for d in DETECTORS:
            hits = [r["detected"][d] for r in sel]
            delays = [r["delay"][d] for r in sel]
            block["detectors"][d] = {
                "detected": int(sum(hits)),
                "detected_at_first_checkpoint": int(sum(h and dl == 1 for h, dl in zip(hits, delays))),
                "mean_delay_censored": float(np.mean(delays)) if sel else np.nan,
            }
        out[name] = block
    missed = [r["true_effect"] for r in reg if not r["detected"]["mcnemar_seq"]]
    caught = [r["true_effect"] for r in reg if r["detected"]["mcnemar_seq"]]
    out["largest_effect_missed_by_mcnemar_seq"] = float(max(missed)) if missed else np.nan
    out["smallest_effect_caught_by_mcnemar_seq"] = float(min(caught)) if caught else np.nan
    return out


def pooled_items(runs: list[RunResult], cfg: MonitorConfig) -> dict:
    """The paired test with main and twin pooled into one 400-item eval (added after the runs)."""
    hits, delays, alarmed = [], [], []
    for r in runs:
        pooled = r.score.reshape(r.score.shape[0], r.score.shape[1], -1)
        det = regression_alerts(pooled, cfg)["mcnemar_seq"]
        if r.fault.kind == "clean":
            alarmed.append(bool(det.any()))
        elif r.fault.kind in REAL_REGRESSIONS:
            hit, delay = detection_delay(det, r.onset_index, affected_evals(r))
            hits.append(hit)
            delays.append(delay)
    return {
        "n_items": int(runs[0].score.shape[-1] * 2),
        "regression_runs": len(hits),
        "detected": int(sum(hits)),
        "mean_delay_censored": float(np.mean(delays)),
        "clean_runs_alarmed": _rate(int(sum(alarmed)), len(alarmed)),
        "min_detectable_drop": stats.minimum_detectable_drop(
            int(runs[0].score.shape[-1] * 2),
            cfg.alpha / ((runs[0].score.shape[0] - cfg.first_look) * len(runs[0].skills) * len(cfg.lags)),
        ),
    }


DROP_SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)  # points of score, i.e. 2 to 40 of 200 items
DROP_AT = 16  # checkpoint index at which the planted drop appears


def planted_drops(runs: list[RunResult], cfg: MonitorConfig, seed: int = 13) -> dict:
    """Power against drops of exactly known size, planted in the clean runs' real results.

    Added after the runs (see Amendments in PLAN.md). For every clean run and every eval
    that scored 90% or more just before checkpoint ``DROP_AT``, a chosen number of items
    that were right at that point are marked wrong from ``DROP_AT`` to the end of the run,
    as if the model had forgotten them. Everything else in the run is real. A drop counts
    as caught if the detector alerts on that eval at ``DROP_AT`` or one of the next three
    checkpoints.
    """
    rng = np.random.default_rng(seed)
    names = ("threshold", "z_seq", "mcnemar_seq", "mcnemar_confirm", "evalue_seq")
    clean = [r for r in runs if r.fault.kind == "clean"]
    n = clean[0].score.shape[-1]
    out: dict = {"drop_at_step": int(clean[0].steps[DROP_AT]), "window": 4, "sizes": {}}
    for size in (0,) + DROP_SIZES:
        hits = {d: [] for d in names}
        first = {d: [] for d in names}
        for r in clean:
            base = r.score[:, :, 0]
            for k in range(base.shape[1]):
                right = np.nonzero(base[DROP_AT - 1, k])[0]
                if right.size < 0.9 * n:
                    continue
                score = base.copy()
                lost = rng.choice(right, size=int(round(size * n / 100)), replace=False)
                score[DROP_AT:, k, lost] = False
                det = regression_alerts(score, cfg)
                for d in names:
                    hits[d].append(bool(det[d][DROP_AT : DROP_AT + 4, k].any()))
                    first[d].append(bool(det[d][DROP_AT, k]))
        out["sizes"][str(size)] = {
            "n_cases": len(hits["z_seq"]),
            "caught_within_window": {d: float(np.mean(v)) for d, v in hits.items()},
            "caught_at_once": {d: float(np.mean(v)) for d, v in first.items()},
        }
    for d in names:
        ok = [s for s in DROP_SIZES if out["sizes"][str(s)]["caught_within_window"][d] >= 0.9]
        out.setdefault("smallest_drop_caught_9_times_in_10", {})[d] = min(ok) if ok else None
    return out


# --- attribution, small evals, health -------------------------------------------------------


def attribution_summary(rows: list[dict]) -> dict:
    table: dict = defaultdict(Counter)
    for r in rows:
        table[r["kind"]][r["attribution"]] += 1
    cats = ("model", "grader", "contamination", "mixed", "none")
    out = {k: {c: int(v[c]) for c in cats} | {"n_runs": int(sum(v.values()))} for k, v in table.items()}
    right = {"corrupt": "model", "remove": "model", "lr_spike": "model", "contaminate": "contamination", "grader_bug": "grader"}
    meas = [r for r in rows if r["kind"] in MEASUREMENT_FAULTS]
    real = [r for r in rows if r["kind"] in REAL_REGRESSIONS]
    out["measurement_faults_blamed_on_eval"] = _rate(sum(r["attribution"] == right[r["kind"]] for r in meas), len(meas))
    out["measurement_faults_blamed_on_model"] = _rate(sum(r["attribution"] in ("model", "mixed") for r in meas), len(meas))
    out["real_regressions_blamed_on_eval"] = _rate(sum(r["attribution"] in ("grader", "contamination", "mixed") for r in real), len(real))
    out["real_regressions_blamed_on_model"] = _rate(sum(r["attribution"] == "model" for r in real), len(real))
    return out


def small_evals(runs: list[RunResult], cfg: MonitorConfig, seed: int = 11) -> dict:
    """Repeat the monitoring on random subsets of the items of each eval."""
    rng = np.random.default_rng(seed)
    n_full = runs[0].score.shape[-1]
    clean = [r for r in runs if r.fault.kind == "clean"]
    out: dict = {"repeats": SUBSET_REPEATS, "sizes": {}}
    for size in SUBSET_SIZES:
        reps = 1 if size == n_full else SUBSET_REPEATS
        acc: dict = defaultdict(list)
        for _ in range(reps):
            idx = np.sort(rng.choice(n_full, size=size, replace=False))
            for r in runs:
                score = r.score[:, :, 0][:, :, idx]
                det = regression_alerts(score, cfg)
                if r.fault.kind == "clean":
                    for d in ("threshold", "z_naive", "mcnemar_seq"):
                        acc[f"clean_fa/{d}"].append(float(det[d].any()))
                        acc[f"clean_alerts/{d}"].append(float(det[d].sum()))
                elif r.fault.kind in REAL_REGRESSIONS:
                    hit, delay = detection_delay(det["mcnemar_seq"], r.onset_index, affected_evals(r))
                    acc[f"power/{r.fault.kind}"].append(float(hit))
                    acc[f"delay/{r.fault.kind}"].append(float(delay))
                    acc["power/all"].append(float(hit))
                    acc["delay/all"].append(float(delay))
        rel = [stats.split_half_reliability(r.score[cfg.first_look :, k, 0][:, :size], n_splits=50)
               for r in clean for k in range(len(r.skills))]
        out["sizes"][str(size)] = {k: float(np.mean(v)) for k, v in sorted(acc.items())} | {
            "min_detectable_drop": stats.minimum_detectable_drop(
                size, cfg.alpha / ((runs[0].score.shape[0] - cfg.first_look) * len(runs[0].skills) * len(cfg.lags))
            ),
            "median_reliability_clean": float(np.nanmedian(rel)),
        }
    return out


def health_summary(runs: list[RunResult], cfg: MonitorConfig) -> dict:
    """Saturation flags at the end of clean runs, checked against the twin set."""
    clean = [r for r in runs if r.fault.kind == "clean"]
    flagged = confirmed = 0
    per_eval: dict = defaultdict(lambda: {"saturated_runs": 0, "reliability": [], "final_score": []})
    w = cfg.saturation_window
    for r in clean:
        sat = saturation_flags(r.score[:, :, 0], cfg)[-1]
        twin_tail = r.score[-w:, :, 1].mean((0, 2))
        for k, name in enumerate(r.skills):
            per_eval[name]["reliability"].append(stats.split_half_reliability(r.score[cfg.first_look :, k, 0]))
            per_eval[name]["final_score"].append(float(r.score[-1, k, 0].mean()))
            if sat[k]:
                flagged += 1
                per_eval[name]["saturated_runs"] += 1
                confirmed += int(twin_tail[k] >= 0.97)
    evals = {}
    for name, v in per_eval.items():
        rel = np.asarray(v["reliability"], dtype=float)
        evals[name] = {
            "saturated_runs": v["saturated_runs"],
            "n_runs": len(clean),
            "median_reliability": float(np.nanmedian(rel)) if np.isfinite(rel).any() else np.nan,
            "mean_final_score": float(np.mean(v["final_score"])),
        }
    return {
        "saturation_flags": flagged,
        "saturation_confirmed_by_twin": _rate(confirmed, flagged),
        "evals": evals,
    }


# --- hypotheses -----------------------------------------------------------------------------


def judge(summary: dict) -> dict:
    """Apply the decision rules written in PLAN.md. Nothing here is a matter of taste."""
    fa = summary["false_alarms"]["real_clean"]["detectors"]
    syn = summary["false_alarms"]["synthetic_null"]["run_level"]
    det = summary["detection"]
    pv = summary["paired_vs_unpaired"]
    att = summary["attribution"]
    small = summary["small_evals"]["sizes"]
    health = summary["health"]
    alpha = summary["config"]["monitor"]["alpha"]
    h = {}

    naive = {d: fa[d]["run_level"]["rate"] for d in ("threshold", "z_naive")}
    h["H1"] = {"supported": all(v > 0.5 for v in naive.values()), "evidence": naive}

    syn_rates = {d: syn[d]["rate"] for d in CORRECTED}
    h["H2a"] = {"supported": all(v <= alpha for v in syn_rates.values()), "evidence": syn_rates}

    mc = fa["mcnemar_seq"]["run_level"]
    h["H2b"] = {"supported": mc["count"] >= 2, "evidence": {"clean_runs_alarmed": mc["count"], "n": mc["n"]}}

    conf = fa["mcnemar_confirm"]["run_level"]
    reg = [r for r in summary["per_run"] if r["kind"] in REAL_REGRESSIONS]
    extra = float(np.mean([r["delay"]["mcnemar_confirm"] - r["delay"]["mcnemar_seq"] for r in reg]))
    h["H2c"] = {
        "supported": conf["count"] <= mc["count"] / 2 and extra <= 1.5,
        "evidence": {"clean_runs_alarmed_confirm": conf["count"], "clean_runs_alarmed_plain": mc["count"], "extra_mean_delay": extra},
    }

    lo = pv["difference_ci95"][0]
    matched_ok = all(m["mcnemar"]["mean_delay"] <= m["z"]["mean_delay"] for m in pv["operating_curve"]["matched"].values())
    h["H3"] = {
        "supported": lo > 0 and matched_ok,
        "evidence": {"mean_difference": pv["mean_difference"], "ci95": pv["difference_ci95"], "paired_no_later_at_matched_rates": matched_ok},
    }

    p = {k: det[k]["detectors"]["mcnemar_seq"] for k in REAL_REGRESSIONS}
    h["H4"] = {
        "supported": p["corrupt"]["detected_at_first_checkpoint"] == det["corrupt"]["n_runs"]
        and p["remove"]["detected"] >= 4
        and p["lr_spike"]["detected"] >= 3,
        "evidence": {
            "corrupt_at_first_checkpoint": p["corrupt"]["detected_at_first_checkpoint"],
            "remove_detected": p["remove"]["detected"],
            "lr_spike_detected": p["lr_spike"]["detected"],
        },
    }

    h["H5"] = {
        "supported": att["grader_bug"]["grader"] == att["grader_bug"]["n_runs"]
        and att["contaminate"]["contamination"] >= 4
        and att["real_regressions_blamed_on_eval"]["count"] == 0,
        "evidence": {
            "grader_bug_right": att["grader_bug"]["grader"],
            "contaminate_right": att["contaminate"]["contamination"],
            "real_regressions_blamed_on_eval": att["real_regressions_blamed_on_eval"]["count"],
        },
    }

    thr = [small[str(s)]["clean_fa/threshold"] for s in SUBSET_SIZES]
    pw = {str(s): small[str(s)]["power/remove"] for s in SUBSET_SIZES}
    h["H6"] = {
        "supported": pw["25"] < 0.5 * pw["200"] and all(a >= b for a, b in zip(thr, thr[1:])),
        "evidence": {"remove_power_by_size": pw, "threshold_clean_fa_by_size": dict(zip(map(str, SUBSET_SIZES), thr))},
    }

    e_extra = float(np.mean([r["delay"]["evalue_seq"] - r["delay"]["mcnemar_seq"] for r in reg]))
    h["H7"] = {
        "supported": fa["evalue_seq"]["false_alerts_per_run"] < fa["mcnemar_seq"]["false_alerts_per_run"] and e_extra <= 1.0,
        "evidence": {
            "false_alerts_per_run_evalue": fa["evalue_seq"]["false_alerts_per_run"],
            "false_alerts_per_run_mcnemar": fa["mcnemar_seq"]["false_alerts_per_run"],
            "extra_mean_delay": e_extra,
        },
    }

    sat = health["saturation_confirmed_by_twin"]
    h["H8"] = {"supported": sat["n"] > 0 and sat["rate"] >= 0.9, "evidence": {"confirmed": sat["count"], "flags": sat["n"]}}
    for v in h.values():
        v["outcome"] = "supported" if v.pop("supported") else "failed"
    return h


def summarise(results_dir: Path, cfg: MonitorConfig = MonitorConfig(), log=print) -> dict:
    runs = load_all(results_dir)
    if not runs:
        raise SystemExit(f"no result files under {results_dir / 'runs'}")
    clean = [r for r in runs if r.fault.kind == "clean"]
    rows = per_run_table(runs, cfg)
    n_ckpt, n_evals, _, n_items = runs[0].score.shape
    summary = {
        "config": {
            "monitor": cfg.to_dict(),
            "n_checkpoints": n_ckpt,
            "n_evals": n_evals,
            "n_items_per_set": n_items,
            "monitored_looks": n_ckpt - cfg.first_look,
            "first_monitored_step": int(runs[0].steps[cfg.first_look]),
            "tests_per_run_paired": (n_ckpt - cfg.first_look) * n_evals * len(cfg.lags),
            "per_test_level_paired": cfg.alpha / ((n_ckpt - cfg.first_look) * n_evals * len(cfg.lags)),
            "evals": list(runs[0].skills),
            "train": runs[0].meta["spec"]["train"],
        },
        "runs": {
            "total": len(runs),
            "by_kind": dict(Counter(r.fault.kind for r in runs)),
            "mean_train_seconds": float(np.mean([r.meta["train_seconds"] for r in runs])),
            "mean_eval_seconds": float(np.mean([r.meta["eval_seconds"] for r in runs])),
            "total_compute_minutes": float(sum(r.meta["train_seconds"] + r.meta["eval_seconds"] for r in runs) / 60),
        },
        "false_alarms": {
            "real_clean": clean_false_alarms(clean, cfg, 0),
            "real_clean_twin_items": clean_false_alarms(clean, cfg, 1),
            "synthetic_null": synthetic_null(cfg),
        },
        "detection": detection_summary(rows),
        "paired_vs_unpaired": paired_vs_unpaired(rows, runs, cfg),
        "by_true_effect": by_true_effect(rows),
        "pooled_items": pooled_items(runs, cfg),
        "planted_drops": planted_drops(runs, cfg),
        "attribution": attribution_summary(rows),
        "small_evals": small_evals(runs, cfg),
        "health": health_summary(runs, cfg),
        "per_run": rows,
    }
    summary["hypotheses"] = judge(summary)
    summary = _round(summary)
    out = results_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    log(f"wrote {out}")
    for name, v in summary["hypotheses"].items():
        log(f"  {name}: {v['outcome']}  {json.dumps(v['evidence'])}")
    return summary
