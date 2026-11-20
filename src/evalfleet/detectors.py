"""Regression detectors and eval-health checks over a whole run.

All functions take item-level results ``score[checkpoint, eval, item]`` (booleans) and
return one boolean alert matrix ``[checkpoint, eval]`` per detector: True where that
detector would raise an alert when that checkpoint arrives. Only checkpoints from
``first_look`` onwards are monitored; earlier ones can still serve as references.

Error-rate accounting for the corrected detectors. A run has ``n_looks`` monitored
checkpoints and ``n_evals`` evals, and each look compares against ``len(lags)`` earlier
checkpoints. Every corrected detector splits one budget ``alpha`` over that whole family
with a union bound, so the chance of any false alert in a run with no regression is at
most ``alpha``, whatever the dependence between looks, lags and evals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import stats

NAIVE = ("threshold", "z_naive")
CORRECTED = ("z_seq", "mcnemar_seq", "mcnemar_confirm", "mcnemar_bh", "evalue_seq")
DETECTORS = NAIVE + CORRECTED

DETECTOR_LABELS = {
    "threshold": "Score-drop threshold",
    "z_naive": "Unpaired z-test, uncorrected",
    "z_seq": "Unpaired z-test, corrected",
    "mcnemar_seq": "Paired exact test, corrected",
    "mcnemar_confirm": "Paired exact test, corrected, confirmed at the next look",
    "mcnemar_bh": "Paired exact test, BH across evals",
    "evalue_seq": "Paired e-value with margin, corrected",
}


@dataclass(frozen=True)
class MonitorConfig:
    alpha: float = 0.05  # run-level false-alarm budget of each detector family
    first_look: int = 8  # index of the first monitored checkpoint
    lags: tuple[int, ...] = (1, 4)  # compare each checkpoint with these earlier ones
    planned_looks: int | None = None  # looks the budget is split over; default: all in the run
    spending: str = "uniform"
    threshold_drop: float = 0.05  # naive detector: alert on a drop of this size or more
    margin: float = 0.02  # e-value detector: drops up to this size are treated as noise
    saturation_level: float = 0.98
    saturation_window: int = 3
    grader_tolerance: float = 0.005  # disagreement rate between graders accepted as normal

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairTables:
    """Paired counts for every (lag, checkpoint, eval). Invalid cells have ``valid`` False."""

    lost: np.ndarray  # right at the reference, wrong now
    gained: np.ndarray  # wrong at the reference, right now
    k_ref: np.ndarray  # number correct at the reference
    k_new: np.ndarray  # number correct now, [checkpoint, eval]
    valid: np.ndarray
    n: int


def pair_tables(score: np.ndarray, cfg: MonitorConfig) -> PairTables:
    score = np.asarray(score, dtype=bool)
    n_ckpt, n_evals, n = score.shape
    shape = (len(cfg.lags), n_ckpt, n_evals)
    lost = np.zeros(shape, dtype=np.int64)
    gained = np.zeros(shape, dtype=np.int64)
    k_ref = np.zeros(shape, dtype=np.int64)
    valid = np.zeros(shape, dtype=bool)
    for j, lag in enumerate(cfg.lags):
        if lag >= n_ckpt:
            continue
        lo, ga = stats.discordant_counts(score[:-lag], score[lag:])
        lost[j, lag:] = lo
        gained[j, lag:] = ga
        k_ref[j, lag:] = score[:-lag].sum(-1)
        valid[j, max(lag, cfg.first_look) :] = True
    return PairTables(lost, gained, k_ref, score.sum(-1), valid, n)


def look_levels(n_ckpt: int, cfg: MonitorConfig) -> np.ndarray:
    """Per-checkpoint share of alpha (zero before ``first_look``)."""
    n_looks = cfg.planned_looks or max(0, n_ckpt - cfg.first_look)
    out = np.zeros(n_ckpt)
    sched = stats.spending_schedule(cfg.alpha, max(n_looks, 1), cfg.spending)
    m = min(n_looks, n_ckpt - cfg.first_look)
    if m > 0:
        out[cfg.first_look : cfg.first_look + m] = sched[:m]
    return out


def regression_pvalues(tables: PairTables) -> dict[str, np.ndarray]:
    """One-sided p-values ``[lag, checkpoint, eval]`` (1 where the comparison is not valid)."""
    p_z = stats.two_proportion_z_pvalue(tables.k_ref, tables.n, tables.k_new[None], tables.n)
    p_mc = stats.mcnemar_exact_pvalue(tables.lost, tables.gained)
    return {
        "z": np.where(tables.valid, p_z, 1.0),
        "mcnemar": np.where(tables.valid, p_mc, 1.0),
    }


def regression_alerts(score: np.ndarray, cfg: MonitorConfig = MonitorConfig()) -> dict[str, np.ndarray]:
    """Alert matrices ``[checkpoint, eval]`` for every regression detector."""
    tables = pair_tables(score, cfg)
    n_ckpt, n_evals, _ = np.shape(score)
    n_lags = len(cfg.lags)
    p = regression_pvalues(tables)
    level = look_levels(n_ckpt, cfg)[:, None]  # alpha available at each look, for the fleet
    monitored = np.arange(n_ckpt)[:, None] >= cfg.first_look
    mean = np.asarray(score, dtype=float).mean(-1)

    out: dict[str, np.ndarray] = {}
    drop = np.zeros_like(mean)
    drop[1:] = mean[:-1] - mean[1:]
    out["threshold"] = (drop >= cfg.threshold_drop - 1e-12) & monitored
    lag1 = cfg.lags.index(1) if 1 in cfg.lags else 0
    out["z_naive"] = (p["z"][lag1] < cfg.alpha) & monitored

    per_test = level / (n_evals * n_lags)  # Bonferroni over evals and lags within a look
    out["z_seq"] = (p["z"] < per_test).any(0)
    out["mcnemar_seq"] = (p["mcnemar"] < per_test).any(0)
    # Persistence rule: alert only when two looks in a row reject. Stricter, so still valid.
    confirm = np.zeros_like(out["mcnemar_seq"])
    confirm[1:] = out["mcnemar_seq"][1:] & out["mcnemar_seq"][:-1]
    out["mcnemar_confirm"] = confirm

    # Benjamini-Hochberg across the fleet at each look, on lag-corrected p-values.
    p_eval = np.minimum(1.0, p["mcnemar"].min(0) * n_lags)
    bh = np.zeros((n_ckpt, n_evals), dtype=bool)
    for t in range(cfg.first_look, n_ckpt):
        if level[t, 0] > 0:
            bh[t] = stats.benjamini_hochberg(p_eval[t], level[t, 0])
    out["mcnemar_bh"] = bh

    # E-values for different lags can be averaged (an average of e-values is an e-value),
    # so the lag family costs nothing here; only looks and evals are corrected for.
    e = stats.betting_evalue(tables.lost, tables.gained, tables.n, cfg.margin)
    e = np.where(tables.valid, e, 1.0).mean(0)
    with np.errstate(divide="ignore"):
        out["evalue_seq"] = e >= np.where(level > 0, n_evals / level, np.inf)
    return out


# --- eval health ----------------------------------------------------------------------------


def saturation_flags(score: np.ndarray, cfg: MonitorConfig = MonitorConfig()) -> np.ndarray:
    """True where the eval scored at or above the ceiling level on the last few checkpoints."""
    mean = np.asarray(score, dtype=float).mean(-1)
    high = mean >= cfg.saturation_level
    out = np.zeros_like(high)
    w = cfg.saturation_window
    for t in range(w - 1, high.shape[0]):
        out[t] = high[t - w + 1 : t + 1].all(0)
    return out


def twin_gap_alerts(
    score: np.ndarray, twin: np.ndarray, cfg: MonitorConfig = MonitorConfig()
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contamination check: is the main set pulling ahead of its fresh twin?

    Main and twin are independent draws of items from one distribution, so with no
    contamination any item-level quantity has the same rate in both. Two such quantities
    are compared with Fisher's exact test at every look:

    * level - items answered correctly now;
    * gain  - items answered correctly now that were wrong just before monitoring began.

    The gain test ignores items that were already right, which makes it more sensitive to
    a slow leak. The budget is split over looks, evals and the two tests. Returns
    (alerts, level p-values, gain p-values).
    """
    score = np.asarray(score, dtype=bool)
    twin = np.asarray(twin, dtype=bool)
    n_main, n_twin = score.shape[-1], twin.shape[-1]
    p_level = stats.fisher_exact_greater(score.sum(-1), n_main, twin.sum(-1), n_twin)
    base = max(0, cfg.first_look - 1)
    p_gain = stats.fisher_exact_greater(
        (score & ~score[base]).sum(-1), n_main, (twin & ~twin[base]).sum(-1), n_twin
    )
    level = look_levels(score.shape[0], cfg)[:, None] / (score.shape[1] * 2)
    return (p_level < level) | (p_gain < level), p_level, p_gain


def grader_alerts(
    score: np.ndarray, regrade: np.ndarray, cfg: MonitorConfig = MonitorConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Grader self-consistency: do two graders disagree on the same outputs more than usual?

    Exact binomial test of the disagreement count against the tolerated rate, corrected
    over looks and evals. Returns (alerts, disagreement counts).
    """
    disagree = (np.asarray(score, dtype=bool) != np.asarray(regrade, dtype=bool)).sum(-1)
    p = stats.binomial_tail_pvalue(disagree, np.shape(score)[-1], cfg.grader_tolerance)
    level = look_levels(disagree.shape[0], cfg)[:, None] / disagree.shape[1]
    return p < level, disagree


def first_alert(alerts: np.ndarray, after: int = -1) -> np.ndarray:
    """Index of the first alert per eval strictly after checkpoint ``after`` (-1 if none)."""
    a = np.asarray(alerts, dtype=bool).copy()
    a[: after + 1] = False
    idx = a.argmax(0)
    return np.where(a.any(0), idx, -1)


def detection_delay(alerts: np.ndarray, onset: int, evals: list[int]) -> tuple[bool, int]:
    """(detected, delay in checkpoints). The first affected checkpoint has delay 1.

    A fault that is never flagged gets the censored delay: one more than the number of
    checkpoints that were left in the run.
    """
    first = first_alert(alerts[:, evals], after=onset)
    hits = first[first >= 0]
    if hits.size == 0:
        return False, alerts.shape[0] - onset
    return True, int(hits.min()) - onset
