"""Every figure in the README and the report, drawn from results/summary.json and the run files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .detectors import MonitorConfig, first_alert, grader_alerts, regression_alerts, twin_gap_alerts  # noqa: E402
from .evaluate import load_results  # noqa: E402
from .study import SUBSET_SIZES  # noqa: E402

BLUE, RED, GREY, GREEN = "#2F5D9B", "#BA3D1A", "#5A6678", "#3F8F6B"
INK = "#1d2430"

SHORT = {
    "threshold": "Drop of 5 points\n(no statistics)",
    "z_naive": "Unpaired z-test\nuncorrected",
    "z_seq": "Unpaired z-test\ncorrected",
    "mcnemar_seq": "Paired exact test\ncorrected",
    "mcnemar_confirm": "Paired exact test\ncorrected + confirmed",
    "mcnemar_bh": "Paired exact test\nBH across evals",
    "evalue_seq": "Paired e-value\n2-point margin",
}
KIND_LABEL = {
    "clean": "No fault",
    "corrupt": "Training targets corrupted",
    "remove": "Training data removed",
    "lr_spike": "Learning-rate spike",
    "contaminate": "Eval items leaked into training",
    "grader_bug": "Grader bug",
}


# One example run per panel of the trajectories figure.
DEFAULT_TRAJECTORY_RUNS = ["corrupt-04", "remove-00", "lr_spike-00", "contaminate-01", "grader_bug-00", "clean-00"]
TRAJECTORY_RUNS = DEFAULT_TRAJECTORY_RUNS


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#9aa1ad",
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e3e0d9",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "text.color": INK,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def fig_headline(summary: dict, out: Path) -> None:
    fa = summary["false_alarms"]
    order = ["threshold", "z_naive", "z_seq", "mcnemar_seq", "mcnemar_confirm", "mcnemar_bh", "evalue_seq"]
    real = [fa["real_clean"]["detectors"][d]["run_level"] for d in order]
    syn = [fa["synthetic_null"]["run_level"][d] for d in order]
    n_clean = fa["real_clean"]["n_runs"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})
    y = np.arange(len(order))[::-1]
    h = 0.36
    r = np.array([v["rate"] for v in real])
    lo = np.array([v["ci95"][0] for v in real])
    hi = np.array([v["ci95"][1] for v in real])
    ax.barh(y + h / 2, r, h, color=BLUE, label=f"{n_clean} real clean runs (95% interval)")
    ax.errorbar(r, y + h / 2, xerr=[r - lo, hi - r], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    s = np.array([v["rate"] for v in syn])
    ax.barh(y - h / 2, s, h, color=GREY, label=f"{fa['synthetic_null']['n_runs']:,} simulated runs with nothing changing")
    ax.axvline(0.05, color=RED, linestyle="--", linewidth=1.2, label="Budget of the corrected detectors (5%)")
    for yi, ri, hi_i, si in zip(y, r, hi, s):
        ax.text(hi_i + 0.015, yi + h / 2, f"{100 * ri:.0f}%", va="center", fontsize=9)
        ax.text(max(si, 0.05) + 0.015, yi - h / 2, f"{100 * si:.1f}%", va="center", fontsize=9, color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[d] for d in order], fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of fault-free runs with at least one false alert")
    ax.set_title("False alarms")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=1, fontsize=9, handlelength=1.6)

    det = summary["detection"]
    kinds = ["corrupt", "remove", "lr_spike"]
    series = [("z_seq", RED, "Unpaired z-test, corrected"), ("mcnemar_seq", BLUE, "Paired exact test, corrected"),
              ("mcnemar_confirm", GREY, "Paired exact test, confirmed"), ("evalue_seq", GREEN, "Paired e-value, 2-point margin")]
    x = np.arange(len(kinds))
    w = 0.21
    top = 0.0
    for j, (d, colour, label) in enumerate(series):
        m = np.array([det[k]["detectors"][d]["mean_delay_censored"] for k in kinds])
        ci = np.array([det[k]["detectors"][d]["mean_delay_ci95"] for k in kinds])
        pos = x + (j - 1.5) * w
        bx.bar(pos, m, w * 0.92, color=colour, label=label)
        bx.errorbar(pos, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for p, mi, hi_i in zip(pos, m, ci[:, 1]):
            bx.text(p, hi_i + 0.2, f"{mi:.1f}", ha="center", fontsize=7.5)
        top = max(top, ci[:, 1].max())
    bx.set_xticks(x)
    short = {"corrupt": "Targets corrupted", "remove": "Data removed", "lr_spike": "Learning-rate spike"}
    bx.set_xticklabels([f"{short[k]}\n({det[k]['n_runs']} runs)" for k in kinds], fontsize=9)
    bx.set_ylabel("Mean checkpoints from fault to first alert")
    bx.set_ylim(0, top + 1.2)
    bx.set_title("Detection delay (lower is better)")
    bx.grid(axis="x", visible=False)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9, handlelength=1.4)
    fig.savefig(out / "headline.png")
    plt.close(fig)


def fig_trajectories(results_dir: Path, summary: dict, out: Path) -> None:
    cfg = MonitorConfig(**{**summary["config"]["monitor"], "lags": tuple(summary["config"]["monitor"]["lags"])})
    picks = TRAJECTORY_RUNS
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharey=True, gridspec_kw={"hspace": 0.5, "wspace": 0.08})
    for ax, run_id in zip(axes.ravel(), picks):
        run = load_results(results_dir / "runs" / f"{run_id}.npz")
        kind = run.fault.kind
        if run.fault.skill is not None:
            k = run.skills.index(run.fault.skill)
        elif kind == "lr_spike":
            twin = run.regrade[:, :, 1].mean(-1)
            k = int((twin[run.onset_index] - twin[run.onset_index + 1 :].min(0)).argmax())
        else:
            k = run.skills.index("sort")
        main = 100 * run.score[:, k, 0].mean(-1)
        steps = run.steps
        ax.axvspan(steps[0], steps[cfg.first_look], color="#f1eee7", lw=0)
        if kind == "grader_bug":
            ax.plot(steps, 100 * run.regrade[:, k, 0].mean(-1), color=GREEN, lw=1.4, ls="--")
        else:
            ax.plot(steps, 100 * run.score[:, k, 1].mean(-1), color=GREY, lw=1.3, ls="--")
        ax.plot(steps, main, color=BLUE, lw=1.9)
        if kind != "clean":
            ax.axvline(run.fault.onset_step, color=RED, lw=1.1, ls=":")
        if kind in ("corrupt", "remove", "lr_spike"):
            alerts = regression_alerts(run.score[:, :, 0], cfg)["mcnemar_seq"]
        elif kind == "contaminate":
            alerts = twin_gap_alerts(run.score[:, :, 0], run.score[:, :, 1], cfg)[0]
        elif kind == "grader_bug":
            alerts = grader_alerts(run.score[:, :, 0], run.regrade[:, :, 0], cfg)[0]
        else:
            alerts = regression_alerts(run.score[:, :, 0], cfg)["mcnemar_seq"]
        t = int(first_alert(alerts, after=run.onset_index if run.onset_index is not None else -1)[k])
        if t >= 0:
            ax.plot([steps[t]], [main[t]], marker="o", ms=7, mfc=RED, mec="white", mew=1.2, ls="none", zorder=5, clip_on=False)
        ax.set_title(f"{KIND_LABEL[kind]}: {run.skills[k]}", fontsize=10)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(steps[0], steps[-1])
        ax.set_xticks([400, 800, 1200, 1600, 2000, 2400])
        ax.tick_params(labelsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score (%)")
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=BLUE, lw=1.9, label="Main item set (what the monitor watches)"),
        Line2D([], [], color=GREY, lw=1.3, ls="--", label="Fresh twin item set"),
        Line2D([], [], color=GREEN, lw=1.4, ls="--", label="Main set, reference grader (grader-bug panel)"),
        Line2D([], [], color=RED, lw=1.1, ls=":", label="Fault starts"),
        Line2D([], [], marker="o", ms=7, mfc=RED, mec="white", ls="none", label="First alert after the fault"),
        Patch(color="#f1eee7", label="Warm-up, not monitored"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.savefig(out / "trajectories.png")
    plt.close(fig)


def fig_operating_curve(summary: dict, out: Path) -> None:
    curve = summary["paired_vs_unpaired"]["operating_curve"]
    n_clean = summary["false_alarms"]["real_clean"]["n_runs"]
    n_reg = summary["paired_vs_unpaired"]["n_runs"]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for test, colour, label in (("z", RED, "Unpaired z-test"), ("mcnemar", BLUE, "Paired exact test")):
        fa = np.array(curve[test]["clean_run_false_alarm"])
        delay = np.array(curve[test]["mean_delay"])
        ax.plot(fa, delay, color=colour, lw=1.8, marker="o", ms=3.5, label=label)
    ax.set_xlabel(f"Share of the {n_clean} clean runs with a false alert")
    ax.set_ylabel(f"Mean detection delay over {n_reg} regression runs\n(checkpoints)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(0, None)
    ax.set_title("Delay against false alarms as the test threshold is varied")
    ax.legend(loc="upper right")
    fig.savefig(out / "operating_curve.png")
    plt.close(fig)


def fig_small_evals(summary: dict, out: Path) -> None:
    sizes = summary["small_evals"]["sizes"]
    xs = list(SUBSET_SIZES)
    pos = np.arange(len(xs))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"wspace": 0.28})
    for kind, colour, marker in (("corrupt", BLUE, "o"), ("remove", RED, "s"), ("lr_spike", GREY, "^")):
        ax.plot(pos, [100 * sizes[str(n)][f"power/{kind}"] for n in xs], color=colour, marker=marker, lw=1.8, label=KIND_LABEL[kind])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Faults flagged before the run ends (%)")
    ax.set_title("Paired exact test: power")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5, handlelength=1.6, columnspacing=1.2)
    for name, colour, marker, label in (
        ("threshold", RED, "s", "Drop of 5 points (no statistics)"),
        ("z_naive", GREY, "^", "Unpaired z-test, uncorrected"),
        ("mcnemar_seq", BLUE, "o", "Paired exact test, corrected"),
    ):
        bx.plot(pos, [100 * sizes[str(n)][f"clean_fa/{name}"] for n in xs], color=colour, marker=marker, lw=1.8, label=label)
    bx.set_ylim(0, 105)
    bx.set_ylabel("Clean runs with a false alert (%)")
    bx.set_title("False alarms on clean runs")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5, handlelength=1.6, columnspacing=1.2)
    for a in (ax, bx):
        a.set_xticks(pos)
        a.set_xticklabels([str(n) for n in xs])
        a.set_xlabel("Items per eval")
    fig.savefig(out / "small_evals.png")
    plt.close(fig)


def fig_effect(summary: dict, out: Path) -> None:
    """Each regression run: how much the model really changed against how soon it was flagged."""
    rows = [r for r in summary["per_run"] if r["kind"] in ("corrupt", "remove", "lr_spike")]
    floor = summary["by_true_effect"]["effect_floor"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    markers = {"corrupt": "o", "remove": "s", "lr_spike": "^"}
    top = max(r["delay"]["mcnemar_seq"] for r in rows) + 1
    for kind, marker in markers.items():
        for hit, face, colour in ((True, BLUE, BLUE), (False, "white", RED)):
            sel = [r for r in rows if r["kind"] == kind and r["detected"]["mcnemar_seq"] == hit]
            if not sel:
                continue
            ax.scatter(
                [max(100 * r["true_effect"], 0.08) for r in sel],
                [r["delay"]["mcnemar_seq"] if hit else top for r in sel],
                marker=marker, s=62, facecolors=face, edgecolors=colour, linewidths=1.6, zorder=3,
                label=f"{KIND_LABEL[kind]}, {'flagged' if hit else 'never flagged'}",
            )
    ax.axvline(100 * floor, color=GREY, lw=1, ls="--", zorder=1)
    ax.text(100 * floor * 1.08, top - 2.2, f"{100 * floor:.0f}-point\nfloor", color=GREY, fontsize=8.5, va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.06, 130)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1 or less", "1", "10", "100"])
    ax.set_yticks(list(range(0, int(top), 2)) + [top])
    ax.set_yticklabels([str(v) for v in range(0, int(top), 2)] + ["never"])
    ax.set_ylim(0, top + 1)
    ax.set_xlabel("True drop on the twin item set, reference grader (points)")
    ax.set_ylabel("Checkpoints from fault to first alert")
    ax.set_title("Paired exact test: what was caught depends on how much the model really moved")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5)
    fig.savefig(out / "effect_vs_detection.png")
    plt.close(fig)


def fig_planted(summary: dict, out: Path) -> None:
    planted = summary["planted_drops"]
    sizes = [int(k) for k in planted["sizes"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for name, colour, marker, label in (
        ("z_seq", RED, "s", "Unpaired z-test, corrected"),
        ("mcnemar_seq", BLUE, "o", "Paired exact test, corrected"),
        ("evalue_seq", GREEN, "^", "Paired e-value, 2-point margin"),
        ("mcnemar_confirm", GREY, "D", "Paired exact test, confirmed"),
    ):
        ax.plot(sizes, [100 * planted["sizes"][str(k)]["caught_within_window"][name] for k in sizes],
                color=colour, marker=marker, ms=5, lw=1.8, label=label)
    n_cases = planted["sizes"]["0"]["n_cases"]
    ax.set_xlabel("Size of the planted drop (points of score; 200 items per eval)")
    ax.set_ylabel(f"Drops caught within 4 checkpoints (%)\n{n_cases} planted cases per size")
    ax.set_xticks(sizes)
    ax.set_ylim(-3, 104)
    ax.set_title("Power against planted drops of known size")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out / "planted_drops.png")
    plt.close(fig)


def fig_attribution(summary: dict, out: Path) -> None:
    att = summary["attribution"]
    kinds = ["clean", "corrupt", "remove", "lr_spike", "contaminate", "grader_bug"]
    cats = [("model", "Model\nregression"), ("grader", "Eval fault:\ngrader"), ("contamination", "Eval fault:\ncontamination"),
            ("mixed", "Model and\neval"), ("none", "No\nfinding")]
    m = np.array([[att[k][c] for c, _ in cats] for k in kinds], dtype=float)
    share = m / m.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    from matplotlib.colors import LinearSegmentedColormap

    ax.imshow(share, cmap=LinearSegmentedColormap.from_list("b", ["#ffffff", BLUE]), vmin=0, vmax=1, aspect="auto")
    for i in range(len(kinds)):
        for j in range(len(cats)):
            if m[i, j]:
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=11,
                        color="white" if share[i, j] > 0.55 else INK, fontweight="bold")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c for _, c in cats], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([f"{KIND_LABEL[k]} ({int(m[i].sum())} runs)" for i, k in enumerate(kinds)], fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(cats)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(kinds)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="#dcd8cf", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Number of runs by the monitor's verdict on the affected eval\n(clean and learning-rate runs: any eval)", fontsize=9, labelpad=10)
    fig.savefig(out / "attribution.png")
    plt.close(fig)


def make_figures(results_dir: Path, out: Path, log=print) -> None:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results_dir / "summary.json").read_text())
    fig_headline(summary, out)
    fig_trajectories(results_dir, summary, out)
    fig_operating_curve(summary, out)
    fig_small_evals(summary, out)
    fig_attribution(summary, out)
    fig_effect(summary, out)
    fig_planted(summary, out)
    for p in sorted(out.glob("*.png")):
        log(f"wrote {p}")
