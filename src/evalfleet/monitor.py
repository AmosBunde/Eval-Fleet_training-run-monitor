"""The monitor: turn item-level results of a run into alerts, health flags and verdicts.

The monitor takes three arrays, each ``[checkpoint, eval, item]``:

* ``score``   - item-level results on the main item set (required);
* ``twin``    - results on a fresh twin item set of the same eval (optional; enables the
  contamination check);
* ``regrade`` - the same outputs as ``score`` graded by a second grader (optional; enables
  the grader self-consistency check and lets a regression be confirmed on regraded scores).

A regression alert says the score fell. The verdict says who is to blame: the model, or
the eval.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import stats
from .detectors import (
    MonitorConfig,
    grader_alerts,
    look_levels,
    pair_tables,
    regression_alerts,
    regression_pvalues,
    saturation_flags,
    twin_gap_alerts,
)

PRIMARY = "mcnemar_seq"

VERDICT_TEXT = {
    "ok": "No finding",
    "model_regression": "Model regression",
    "grader_fault": "Eval fault: grader",
    "contamination": "Eval fault: contamination",
}


@dataclass
class Alert:
    kind: str  # "regression", "grader" or "contamination"
    eval: str
    index: int  # checkpoint index at which the alert is raised
    step: int
    text: str
    detail: dict = field(default_factory=dict)


@dataclass
class EvalHealth:
    eval: str
    saturated_from: int | None  # step from which the eval sat at its ceiling, if it did
    reliability: float  # split-half reliability across monitored checkpoints
    min_detectable_drop: float  # smallest one-step drop the paired test can flag
    final_score: float
    verdict: tuple[str, ...]


@dataclass
class MonitorReport:
    evals: tuple[str, ...]
    steps: np.ndarray
    config: MonitorConfig
    mean: np.ndarray  # [checkpoint, eval] main score
    lo: np.ndarray  # Wilson interval
    hi: np.ndarray
    twin_mean: np.ndarray | None
    regrade_mean: np.ndarray | None
    alerts: list[Alert]
    health: list[EvalHealth]
    detector_alerts: dict[str, np.ndarray]  # every regression detector, for comparison
    n_items: int

    def verdicts(self) -> dict[str, tuple[str, ...]]:
        return {h.eval: h.verdict for h in self.health}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _explain_regression(name, steps, t, lag, lost, gained, n, p, level, twin_change) -> str:
    drop = (lost - gained) / n
    text = (
        f"{name}: the score fell {_pct(drop)} points between step {steps[t - lag]} and step "
        f"{steps[t]}. {lost} of {n} items went from right to wrong and {gained} went from "
        f"wrong to right. If the model had not changed on this eval, a split this lopsided "
        f"would have probability {p:.1e}; the alert level after correcting for every look "
        f"and eval is {level:.1e}."
    )
    if twin_change is not None:
        if twin_change <= -0.5 * drop:
            text += (
                f" The fresh twin set fell too ({_pct(-twin_change)} points), so this is not "
                f"specific to these items."
            )
        else:
            text += (
                f" The fresh twin set did not follow (change of {_pct(twin_change)} points)."
            )
    return text


def monitor_run(
    score: np.ndarray,
    steps,
    evals,
    twin: np.ndarray | None = None,
    regrade: np.ndarray | None = None,
    config: MonitorConfig = MonitorConfig(),
) -> MonitorReport:
    """Run every detector and health check over one run and explain what was found."""
    score = np.asarray(score, dtype=bool)
    steps = np.asarray(steps)
    evals = tuple(evals)
    n_ckpt, n_evals, n = score.shape
    if len(evals) != n_evals or len(steps) != n_ckpt:
        raise ValueError("steps and evals must match the first two axes of score")
    mean = score.mean(-1)
    lo, hi = stats.wilson_interval(score.sum(-1), n)
    twin_mean = None if twin is None else np.asarray(twin, dtype=bool).mean(-1)
    regrade_mean = None if regrade is None else np.asarray(regrade, dtype=bool).mean(-1)

    det = regression_alerts(score, config)
    alerts: list[Alert] = []
    labels: dict[str, set[str]] = defaultdict(set)

    # Regression alerts from the primary detector, each explained from its paired counts.
    tables = pair_tables(score, config)
    p_mc = regression_pvalues(tables)["mcnemar"]
    level = look_levels(n_ckpt, config) / (n_evals * len(config.lags))
    for t, k in zip(*np.nonzero(det[PRIMARY])):
        j = int(p_mc[:, t, k].argmin())
        lag = config.lags[j]
        twin_change = None if twin_mean is None else float(twin_mean[t, k] - twin_mean[t - lag, k])
        lost, gained = int(tables.lost[j, t, k]), int(tables.gained[j, t, k])
        alerts.append(
            Alert(
                "regression",
                evals[k],
                int(t),
                int(steps[t]),
                _explain_regression(
                    evals[k], steps, t, lag, lost, gained, n, float(p_mc[j, t, k]), float(level[t]), twin_change
                ),
                {"lag": lag, "lost": lost, "gained": gained, "p": float(p_mc[j, t, k])},
            )
        )

    # Who is to blame? A regression counts against the model only if it survives regrading.
    confirm = det[PRIMARY] if regrade is None else regression_alerts(regrade, config)[PRIMARY]
    for k in np.nonzero(confirm.any(0))[0]:
        labels[evals[k]].add("model_regression")

    if regrade is not None:
        g_alert, disagree = grader_alerts(score, regrade, config)
        for k in np.nonzero(g_alert.any(0))[0]:
            t = int(g_alert[:, k].argmax())
            labels[evals[k]].add("grader_fault")
            alerts.append(
                Alert(
                    "grader",
                    evals[k],
                    t,
                    int(steps[t]),
                    f"{evals[k]}: from step {steps[t]} the production grader and the reference "
                    f"grader disagree on {int(disagree[t, k])} of {n} items for the same model "
                    f"outputs (up to {int(disagree[: t, k].max(initial=0))} before). The "
                    f"reference grader gives {_pct(regrade_mean[t, k])}% where the production "
                    f"grader gives {_pct(mean[t, k])}%. Check the grader before reading this "
                    f"eval's score.",
                    {"disagree": int(disagree[t, k])},
                )
            )

    if twin is not None:
        c_alert, p_level, p_gain = twin_gap_alerts(score, twin, config)
        base = max(0, config.first_look - 1)
        for k in np.nonzero(c_alert.any(0))[0]:
            t = int(c_alert[:, k].argmax())
            labels[evals[k]].add("contamination")
            new_main = int((score[t, k] & ~score[base, k]).sum())
            new_twin = int((twin[t, k] & ~twin[base, k]).sum())
            alerts.append(
                Alert(
                    "contamination",
                    evals[k],
                    t,
                    int(steps[t]),
                    f"{evals[k]}: at step {steps[t]} the main item set scores "
                    f"{_pct(mean[t, k])}% but a fresh twin set drawn the same way scores "
                    f"{_pct(twin_mean[t, k])}%. Since step {steps[base]}, {new_main} main items "
                    f"have turned right against {new_twin} twin items. Sampling alone would "
                    f"give a difference this large with probability "
                    f"{float(min(p_level[t, k], p_gain[t, k])):.1e}. The main items have probably "
                    f"leaked into training; the main score no longer measures the skill.",
                    {
                        "p_level": float(p_level[t, k]),
                        "p_gain": float(p_gain[t, k]),
                        "gap": float(mean[t, k] - twin_mean[t, k]),
                    },
                )
            )

    sat = saturation_flags(score, config)
    mde_alpha = float(level[level > 0].max()) if (level > 0).any() else config.alpha
    health = []
    for k, name in enumerate(evals):
        sat_from = None
        if sat[-1, k]:
            run_start = n_ckpt - 1
            while run_start > 0 and sat[run_start - 1, k]:
                run_start -= 1
            sat_from = int(steps[max(0, run_start - config.saturation_window + 1)])
        health.append(
            EvalHealth(
                name,
                sat_from,
                stats.split_half_reliability(score[config.first_look :, k]),
                stats.minimum_detectable_drop(n, mde_alpha),
                float(mean[-1, k]),
                tuple(sorted(labels[name])) or ("ok",),
            )
        )
    alerts.sort(key=lambda a: (a.index, a.eval, a.kind))
    return MonitorReport(
        evals, steps, config, mean, lo, hi, twin_mean, regrade_mean, alerts, health, det, n
    )


def monitor_result(result, config: MonitorConfig = MonitorConfig()) -> MonitorReport:
    """Monitor a ``RunResult`` produced by ``evalfleet evaluate``."""
    return monitor_run(
        result.score[:, :, 0],
        result.steps,
        result.skills,
        twin=result.score[:, :, 1],
        regrade=result.regrade[:, :, 0],
        config=config,
    )


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read item-level results from a long CSV with columns ``step,eval,item,correct``.

    Every eval must have the same items at every step and the same number of items as
    the other evals. Returns ``(score, steps, evals)`` ready for ``monitor_run``.
    """
    cells: dict[tuple[int, str, str], bool] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cells[(int(row["step"]), row["eval"], row["item"])] = row["correct"].strip() in ("1", "True", "true")
    steps = sorted({s for s, _, _ in cells})
    evals = sorted({e for _, e, _ in cells})
    items = {e: sorted({i for _, ev, i in cells if ev == e}) for e in evals}
    sizes = {len(v) for v in items.values()}
    if len(sizes) != 1:
        raise ValueError(f"every eval needs the same number of items, got {sorted(sizes)}")
    score = np.zeros((len(steps), len(evals), sizes.pop()), dtype=bool)
    for a, s in enumerate(steps):
        for b, e in enumerate(evals):
            for c, i in enumerate(items[e]):
                try:
                    score[a, b, c] = cells[(s, e, i)]
                except KeyError:
                    raise ValueError(f"missing result for step {s}, eval {e}, item {i}") from None
    return score, np.asarray(steps), tuple(evals)


def text_summary(report: MonitorReport) -> str:
    """A terminal-friendly summary of a monitor report."""
    lines = [
        f"{len(report.steps)} checkpoints, {len(report.evals)} evals, {report.n_items} items each; "
        f"monitoring from step {report.steps[min(report.config.first_look, len(report.steps) - 1)]}"
    ]
    lines.append(f"{'eval':<10}{'final':>7}  {'reliab.':>7}  verdict")
    for h in report.health:
        verdict = ", ".join(VERDICT_TEXT[v] for v in h.verdict)
        sat = f"  (saturated from step {h.saturated_from})" if h.saturated_from is not None else ""
        lines.append(f"{h.eval:<10}{100 * h.final_score:>6.1f}%  {h.reliability:>7.2f}  {verdict}{sat}")
    if not report.alerts:
        lines.append("No alerts.")
    for a in report.alerts:
        lines.append(f"[{a.kind} @ step {a.step}] {a.text}")
    return "\n".join(lines)
