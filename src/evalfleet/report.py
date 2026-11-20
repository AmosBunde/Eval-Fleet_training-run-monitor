"""A self-contained HTML report for one run: inline CSS and SVG, no external requests."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from .monitor import VERDICT_TEXT, MonitorReport, episodes

BLUE, RED, GREY, GREEN = "#2F5D9B", "#BA3D1A", "#5A6678", "#3F8F6B"

CSS = """
:root { --ink:#1d2430; --soft:#5a6678; --faint:#8a93a3; --rule:#dcd8cf; --paper:#faf8f4;
  --panel:#ffffff; --blue:#2F5D9B; --red:#BA3D1A; --green:#3F8F6B; --wash:#f1eee7; }
* { box-sizing: border-box; }
html { background: var(--paper); }
body { margin: 0 auto; max-width: 1120px; padding: 48px 32px 72px; color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-variant-numeric: tabular-nums; }
h1, h2 { font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; font-weight: 600;
  letter-spacing: -0.01em; margin: 0; }
h1 { font-size: 34px; line-height: 1.15; }
h2 { font-size: 22px; margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--ink); }
.kicker { font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--soft); margin-bottom: 10px; }
.meta { color: var(--soft); margin-top: 10px; }
.lede { font-size: 18px; line-height: 1.5; max-width: 760px; margin: 22px 0 0; }
.note { color: var(--soft); max-width: 760px; margin: 10px 0 0; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; margin-top: 32px;
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.tile { padding: 18px 20px 16px 0; }
.tile + .tile { padding-left: 20px; border-left: 1px solid var(--rule); }
.tile b { display: block; font: 600 32px/1.1 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; }
.tile span { color: var(--soft); font-size: 13px; }
.tile.bad b { color: var(--red); } .tile.warn b { color: var(--blue); } .tile.good b { color: var(--green); }
.alert { display: grid; grid-template-columns: 170px 1fr; gap: 24px; padding: 20px 0; border-bottom: 1px solid var(--rule); }
.alert:first-of-type { margin-top: 12px; border-top: 1px solid var(--rule); }
.alert .what { font-weight: 600; }
.alert .when { color: var(--soft); font-size: 13px; margin-top: 2px; }
.alert p { margin: 0; max-width: 720px; }
.alert .more { color: var(--soft); font-size: 13px; margin-top: 8px; }
.tag { display: inline-block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
  padding: 2px 8px; border-radius: 2px; margin-bottom: 8px; }
.tag.model { background: var(--red); color: #fff; }
.tag.eval { background: var(--ink); color: #fff; }
.tag.ok { color: var(--green); border: 1px solid var(--green); }
.tag.sat { color: var(--soft); border: 1px solid var(--faint); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 26px; color: var(--soft); font-size: 13px; margin: 18px 0 6px; }
.legend svg { vertical-align: -3px; margin-right: 7px; }
.fleet { display: grid; grid-template-columns: repeat(4, 1fr); gap: 28px 24px; margin-top: 14px; }
.card h3 { margin: 0; font-size: 15px; font-weight: 600; display: flex; justify-content: space-between; align-items: baseline; }
.card h3 span { font-weight: 400; color: var(--soft); font-size: 13px; }
.card svg { display: block; width: 100%; height: auto; margin-top: 6px; }
.card .tags { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
.card .tags .tag { margin: 0; }
table { border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }
th { text-align: right; font-weight: 600; font-size: 12px; letter-spacing: 0.04em; color: var(--soft);
  border-bottom: 1px solid var(--ink); padding: 8px 10px; vertical-align: bottom; }
td { text-align: right; padding: 9px 10px; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
th:last-child, td:last-child { text-align: left; }
td.flag { color: var(--red); font-weight: 600; }
.method { columns: 2; column-gap: 40px; margin-top: 16px; color: var(--soft); font-size: 14px; }
.method p { margin: 0 0 12px; break-inside: avoid; }
.method b { color: var(--ink); font-weight: 600; }
footer { margin-top: 56px; color: var(--faint); font-size: 12px; }
@media (max-width: 900px) { .fleet, .tiles { grid-template-columns: repeat(2, 1fr); }
  .tile:nth-child(3) { border-left: 0; padding-left: 0; } .alert { grid-template-columns: 1fr; gap: 8px; } .method { columns: 1; } }
"""

KIND_LABEL = {"regression": "Score regression", "grader": "Grader inconsistency", "contamination": "Contamination"}


def _chart(report: MonitorReport, k: int) -> str:
    """One eval's trajectory as an SVG: interval band, main line, twin line, alert marks."""
    w, h, left, right, top, bottom = 280, 150, 30, 8, 8, 22
    steps = report.steps.astype(float)
    x0, x1 = steps[0], steps[-1]
    span = (x1 - x0) or 1.0

    def X(s: float) -> float:
        return left + (s - x0) / span * (w - left - right)

    def Y(v: float) -> float:
        return top + (1 - v) * (h - top - bottom)

    def path(vals) -> str:
        return " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(zip(steps, vals)))

    name = report.evals[k]
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Score of {escape(name)} by training step">']
    first = min(report.config.first_look, len(steps) - 1)
    if first > 0:
        parts.append(f'<rect x="{left}" y="{top}" width="{X(steps[first]) - left:.1f}" height="{h - top - bottom}" fill="#f1eee7"/>')
    for v in (0, 0.5, 1):
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#dcd8cf" stroke-width="1"/>')
        parts.append(f'<text x="{left - 6}" y="{Y(v) + 3.5:.1f}" text-anchor="end" font-size="10" fill="#8a93a3">{int(v * 100)}</text>')
    parts.append(f'<text x="{left}" y="{h - 6}" font-size="10" fill="#8a93a3">step {int(x0)}</text>')
    parts.append(f'<text x="{w - right}" y="{h - 6}" text-anchor="end" font-size="10" fill="#8a93a3">{int(x1)}</text>')
    band = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, report.hi[:, k]))
    band += " " + " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps[::-1], report.lo[::-1, k]))
    parts.append(f'<polygon points="{band}" fill="{BLUE}" fill-opacity="0.16"/>')
    if report.twin_mean is not None:
        parts.append(f'<path d="{path(report.twin_mean[:, k])}" fill="none" stroke="{GREY}" stroke-width="1.2" stroke-dasharray="3 2.5"/>')
    parts.append(f'<path d="{path(report.mean[:, k])}" fill="none" stroke="{BLUE}" stroke-width="1.8" stroke-linejoin="round"/>')
    seen = set()
    for a in report.alerts:
        if a.eval != name or (a.kind, a.index) in seen:
            continue
        seen.add((a.kind, a.index))
        cx, cy = X(steps[a.index]), Y(report.mean[a.index, k])
        if a.kind == "regression":
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" fill="{RED}" stroke="#fff" stroke-width="1"/>')
        else:
            cy -= 11  # sit above the line, clear of any regression mark at the same checkpoint
            parts.append(
                f'<rect x="{cx - 3.6:.1f}" y="{cy - 3.6:.1f}" width="7.2" height="7.2" fill="#1d2430" stroke="#fff" '
                f'stroke-width="1" transform="rotate(45 {cx:.1f} {cy:.1f})"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _tags(verdict: tuple[str, ...], saturated_from) -> str:
    out = []
    for v in verdict:
        cls = {"ok": "ok", "model_regression": "model"}.get(v, "eval")
        out.append(f'<span class="tag {cls}">{escape(VERDICT_TEXT[v])}</span>')
    if saturated_from is not None:
        out.append(f'<span class="tag sat">Saturated from step {saturated_from}</span>')
    return "".join(out)


def render(report: MonitorReport, run_id: str) -> str:
    cfg = report.config
    verdicts = report.verdicts()
    n_model = sum("model_regression" in v for v in verdicts.values())
    n_eval = sum(any(x in v for x in ("grader_fault", "contamination")) for v in verdicts.values())
    n_sat = sum(h.saturated_from is not None for h in report.health)
    n_ok = sum(v == ("ok",) for v in verdicts.values())
    first_step = int(report.steps[min(cfg.first_look, len(report.steps) - 1)])
    n_looks = max(0, len(report.steps) - cfg.first_look)

    if n_model == 0 and n_eval == 0:
        lede = "No eval shows a regression, and no eval looks broken. Nothing in this run needs attention."
    else:
        bits = []
        if n_model:
            names = ", ".join(e for e, v in verdicts.items() if "model_regression" in v)
            bits.append(f"The model got worse on {names}.")
        if n_eval:
            names = ", ".join(e for e, v in verdicts.items() if any(x in v for x in ("grader_fault", "contamination")))
            bits.append(f"The eval itself looks faulty for {names}; do not read its score as a statement about the model.")
        lede = " ".join(bits)

    html = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Eval fleet report: {escape(run_id)}</title>",
        f"<style>{CSS}</style></head><body>",
        '<div class="kicker">Eval fleet report</div>',
        f"<h1>Run {escape(run_id)}</h1>",
        f'<div class="meta">{len(report.steps)} checkpoints, steps {int(report.steps[0])} to {int(report.steps[-1])} '
        f"&middot; {len(report.evals)} evals &middot; {report.n_items} items each &middot; monitored from step {first_step}</div>",
        f'<p class="lede">{escape(lede)}</p>',
        '<div class="tiles">',
        f'<div class="tile {"bad" if n_model else "good"}"><b>{n_model}</b><span>evals where the model regressed</span></div>',
        f'<div class="tile {"bad" if n_eval else "good"}"><b>{n_eval}</b><span>evals that look faulty themselves</span></div>',
        f'<div class="tile warn"><b>{n_sat}</b><span>evals at their ceiling</span></div>',
        f'<div class="tile"><b>{n_ok}</b><span>evals with no finding</span></div>',
        "</div>",
    ]

    html.append("<h2>What needs attention</h2>")
    eps = episodes(report.alerts)
    if not eps:
        html.append('<p class="note">No alerts were raised at any monitored checkpoint.</p>')
    for a, repeats, last_step in eps:
        blamed = verdicts[a.eval]
        if a.kind == "regression":
            if "model_regression" in blamed:
                tag = '<span class="tag model">Model</span>'
            elif "grader_fault" in blamed:
                tag = '<span class="tag eval">Eval, not model</span>'
            else:
                tag = '<span class="tag model">Model</span>'
        else:
            tag = '<span class="tag eval">Eval fault</span>'
        more = ""
        if repeats:
            if repeats == 1:
                more = f'<div class="more">Raised again at the next checkpoint, step {last_step}.</div>'
            else:
                more = f'<div class="more">Raised again at each of the next {repeats} checkpoints, up to step {last_step}.</div>'
        if a.kind == "regression" and "grader_fault" in blamed and "model_regression" not in blamed:
            more += '<div class="more">The reference grader sees no such drop on the same outputs, so this is attributed to the grader.</div>'
        body = a.text.split(": ", 1)[1]
        html.append(
            f'<div class="alert"><div>{tag}<div class="what">{escape(KIND_LABEL[a.kind])}: {escape(a.eval)}</div>'
            f'<div class="when">first raised at step {a.step}</div></div>'
            f"<div><p>{escape(body[0].upper() + body[1:])}</p>{more}</div></div>"
        )

    html.append("<h2>The fleet</h2>")
    html.append(
        '<div class="legend">'
        f'<span><svg width="26" height="12"><rect y="1" width="26" height="10" fill="{BLUE}" fill-opacity="0.16"/><line x1="0" x2="26" y1="6" y2="6" stroke="{BLUE}" stroke-width="2"/></svg>main item set, with 95% interval</span>'
        f'<span><svg width="26" height="12"><line x1="0" x2="26" y1="6" y2="6" stroke="{GREY}" stroke-width="1.4" stroke-dasharray="3 2.5"/></svg>fresh twin item set</span>'
        f'<span><svg width="12" height="12"><circle cx="6" cy="6" r="3.6" fill="{RED}"/></svg>regression alert</span>'
        '<span><svg width="12" height="12"><rect x="2.4" y="2.4" width="7.2" height="7.2" fill="#1d2430" transform="rotate(45 6 6)"/></svg>eval-fault alert</span>'
        '<span><svg width="14" height="12"><rect width="14" height="12" fill="#f1eee7" stroke="#dcd8cf"/></svg>warm-up, not monitored</span>'
        "</div>"
    )
    html.append('<div class="fleet">')
    for k, hlt in enumerate(report.health):
        html.append(
            f'<div class="card"><h3>{escape(hlt.eval)}<span>{100 * hlt.final_score:.1f}% at the end</span></h3>'
            f'{_chart(report, k)}<div class="tags">{_tags(hlt.verdict, hlt.saturated_from)}</div></div>'
        )
    html.append("</div>")
    html.append('<p class="note">Score in per cent against training step. Every panel uses the same axes.</p>')

    html.append("<h2>Eval health</h2>")
    html.append(
        "<table><thead><tr><th>Eval</th><th>Final score</th><th>Twin set</th><th>Graders disagree<br>(most items)</th>"
        "<th>Split-half<br>reliability</th><th>Smallest drop<br>it can flag</th><th>Ceiling</th><th>Verdict</th></tr></thead><tbody>"
    )
    for k, hlt in enumerate(report.health):
        twin = "n/a" if report.twin_mean is None else f"{100 * report.twin_mean[-1, k]:.1f}%"
        if report.regrade_mean is None:
            dis = "n/a"
        else:
            worst = int(round(np.abs(report.mean[:, k] - report.regrade_mean[:, k]).max() * report.n_items))
            dis = str(worst)
        rel = "n/a" if np.isnan(hlt.reliability) else f"{hlt.reliability:.2f}"
        sat = "no" if hlt.saturated_from is None else f"from step {hlt.saturated_from}"
        verdict = ", ".join(VERDICT_TEXT[v] for v in hlt.verdict)
        cls = "" if hlt.verdict == ("ok",) else ' class="flag"'
        html.append(
            f"<tr><td>{escape(hlt.eval)}</td><td>{100 * hlt.final_score:.1f}%</td><td>{twin}</td><td>{dis}</td>"
            f"<td>{rel}</td><td>{100 * hlt.min_detectable_drop:.1f} pts</td><td>{sat}</td><td{cls}>{escape(verdict)}</td></tr>"
        )
    html.append("</tbody></table>")
    html.append(
        '<p class="note">Reliability is how consistently two random halves of the items rank the monitored checkpoints '
        "(1 is perfect; it is undefined, shown as n/a, when the score never moves). An eval at its ceiling can still show "
        "a regression but can no longer show an improvement.</p>"
    )

    n_tests = n_looks * len(report.evals) * len(cfg.lags)
    html.append("<h2>How to read this</h2>")
    html.append(
        '<div class="method">'
        f"<p><b>Regression alerts.</b> Each monitored checkpoint is compared with the checkpoints {' and '.join(str(l) for l in cfg.lags)} "
        "back, on the same items. Only items whose result flipped carry evidence: under no change a flip is as likely to be a loss as a gain "
        "(exact McNemar test).</p>"
        f"<p><b>Repeated looks.</b> The run has {n_looks} monitored checkpoints and {len(report.evals)} evals, so {n_tests} tests. One budget of "
        f"{cfg.alpha:g} is split evenly over all of them. If the model never changed, the chance of any regression alert in the whole run is at most {cfg.alpha:g}.</p>"
        "<p><b>What the guarantee does not cover.</b> A checkpoint that is really a little worse than the last one is a real change, "
        "even if it heals by itself. Such dips are flagged. An alert means the items say the model changed; it does not say why.</p>"
        "<p><b>Model or eval.</b> A regression is blamed on the model only if it also shows when the same outputs are graded by a second, "
        "independent grader. If the two graders start to disagree, the grader is blamed. If the main items pull ahead of a fresh twin set "
        "drawn the same way, the eval is marked contaminated.</p>"
        "</div>"
    )
    html.append("<footer>Generated by evalfleet. Self-contained file: no scripts, no external requests.</footer>")
    html.append("</body></html>")
    return "\n".join(html)


def write_report(report: MonitorReport, run_id: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(report, run_id))
    return out
