"""Standalone HTML dashboard report.

Produces a single self-contained file (inline CSS, no external assets, all
dynamic values HTML-escaped) suitable for direct viewing, attaching to CI
artifacts, or publishing through GitHub Pages.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

from dqfs.errors import ReportError

logger = logging.getLogger(__name__)

_STATUS_CLASS = {"pass": "ok", "warn": "warn", "fail": "bad", "n/a": "na"}
_VERDICT_CLASS = {"PASS": "ok", "WARN": "warn", "FAIL": "bad"}


def _esc(value: Any) -> str:
    """Escape any value for safe HTML interpolation."""
    return html.escape(str(value if value is not None else "-"))


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _gauge_svg(dqs: float, verdict: str) -> str:
    """Circular DQS gauge (SVG, no external assets)."""
    color = {"PASS": "#22c55e", "WARN": "#f59e0b", "FAIL": "#ef4444"}.get(verdict, "#64748b")
    radius = 70.0
    circumference = 2 * 3.141592653589793 * radius
    filled = circumference * (max(0.0, min(100.0, dqs)) / 100.0)
    return (
        '<svg viewBox="0 0 180 180" width="180" height="180" role="img" '
        f'aria-label="DQS {dqs:.1f}">'
        '<circle cx="90" cy="90" r="70" fill="none" stroke="#1e293b" stroke-width="14"/>'
        f'<circle cx="90" cy="90" r="70" fill="none" stroke="{color}" stroke-width="14"'
        ' stroke-linecap="round" transform="rotate(-90 90 90)"'
        f' stroke-dasharray="{filled:.1f} {circumference:.1f}"/>'
        '<text x="90" y="85" text-anchor="middle" fill="#e2e8f0" font-size="40" '
        f'font-weight="700">{dqs:.1f}</text>'
        '<text x="90" y="112" text-anchor="middle" fill="#94a3b8" font-size="13">'
        "DQS / 100</text></svg>"
    )


def _factor_cards(factors: dict[str, Any]) -> str:
    cards = []
    for name, info in factors.items():
        score = info.get("score")
        status = info.get("status", "n/a")
        cls = _STATUS_CLASS.get(status, "na")
        bar = (score or 0.0) * 100.0
        weight = info.get("weight")
        weight_html = (
            f'<span class="chip">weight {weight * 100:.1f}%</span>'
            if weight is not None
            else ""
        )
        details = info.get("details") or {}
        detail_bits = []
        for key in ("missing_cells", "invalid_cells", "duplicate_rows",
                    "outlier_cells", "format_mismatch_cells", "median_age_days"):
            if key in details:
                detail_bits.append(f"{key.replace('_', ' ')}: {_esc(details[key])}")
        detail_line = " &middot; ".join(detail_bits) or "&nbsp;"
        cards.append(
            f'<div class="card">'
            f'<div class="card-head"><span class="name">{_esc(name)}</span>'
            f'<span class="chip {cls}">{_esc(status)}</span></div>'
            f'<div class="bar"><div class="fill {cls}" style="width:{bar:.1f}%"></div></div>'
            f'<div class="score-line">score {_esc(score)}</div>'
            f'<div class="meta">{weight_html}'
            f'<span class="chip">IC {_esc(info.get("ic"))}</span>'
            f'<span class="chip">deep imp. {_esc(info.get("deep_importance"))}</span></div>'
            f'<div class="detail">{detail_line}</div>'
            f"</div>"
        )
    return "\n".join(cards)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _gates_table(gates: list[dict[str, Any]]) -> str:
    rows = []
    for g in gates:
        cls = _STATUS_CLASS.get(g.get("status", "n/a"), "na")
        rows.append([
            g.get("name"), f'<span class="chip {cls}">{_esc(g.get("status"))}</span>',
            g.get("value"), g.get("threshold"),
        ])
    return _table(["gate", "status", "value", "thresholds"], rows)


def _columns_table(columns: list[dict[str, Any]]) -> str:
    rows = [
        [c.get("column"), c.get("kind"), _pct(c.get("missing_ratio")),
         _pct(c.get("invalid_ratio")), _pct(c.get("outlier_ratio")),
         c.get("deep_importance")]
        for c in columns
    ]
    return _table(
        ["column", "kind", "missing", "invalid", "outliers", "deep importance"], rows
    )


def _worst_rows_table(worst: list[dict[str, Any]]) -> str:
    rows = []
    for w in worst:
        contrib = ", ".join(f"{k}: {v}" for k, v in (w.get("contributions") or {}).items())
        rows.append([w.get("row_index"), w.get("badness"), contrib])
    return _table(["row", "badness", "factor contributions"], rows)


def _model_card_html(card: dict[str, Any]) -> str:
    if not card.get("enabled", False):
        reason = _esc(card.get("reason", "disabled"))
        return (
            f'<p class="muted">Deep ensemble disabled ({reason}). '
            "Weights fall back to equal-weight floors.</p>"
        )
    arch = card.get("architecture", {})
    rows = []
    for branch, meta in arch.items():
        rows.append([branch, meta.get("type"), meta.get("hidden"), meta.get("parameters")])
    den = card.get("denoising", {})
    bits = [
        f"epochs: {_esc(den.get('epochs'))}",
        f"corruption rate: {_esc(den.get('corruption_rate'))}",
        f"final loss (attention): {_esc(den.get('final_loss_attention'))}",
        f"final loss (MLP): {_esc(den.get('final_loss_mlp'))}",
        f"training rows: {_esc(card.get('training_rows'))}",
        f"runtime: {_esc(card.get('runtime_seconds'))}s",
        f"seed: {_esc(card.get('seed'))}",
    ]
    meta_html = ""
    if card.get("meta_ridge"):
        mr = card["meta_ridge"]
        sigs = _esc(", ".join(mr.get("signals", [])))
        meta_html = (
            f'<p class="detail">meta-learner: ridge over [{sigs}] '
            f'- R&sup2; {_esc(mr.get("r_squared"))} '
            f'coefficients {_esc(mr.get("coefficients"))}</p>'
        )
    supervised = "supervised target: " + _esc(card.get("target", "-")) \
        if card.get("supervised") else "unsupervised"
    return (
        _table(["branch", "type", "hidden", "params"], rows)
        + f'<p class="detail">{" &middot; ".join(bits)}</p>'
        + f'<p class="detail">{supervised}</p>' + meta_html
    )


_CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box;margin:0}
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
background:#0b1220;color:#e2e8f0;padding:32px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;letter-spacing:.3px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:1.4px;color:#94a3b8;
margin:34px 0 12px}
.muted{color:#94a3b8}
.detail{color:#94a3b8;font-size:13px;margin-top:6px}
header{display:flex;justify-content:space-between;align-items:center;
border-bottom:1px solid #1e293b;padding-bottom:20px}
.hero{display:flex;gap:28px;align-items:center;margin:24px 0}
.hero .kv{display:grid;grid-template-columns:auto auto;gap:4px 18px;font-size:14px}
.hero .kv dt{color:#94a3b8}
.badge{padding:4px 14px;border-radius:999px;font-weight:700;font-size:14px}
.badge.ok{background:#052e16;color:#4ade80;border:1px solid #166534}
.badge.warn{background:#422006;color:#fbbf24;border:1px solid #92400e}
.badge.bad{background:#450a0a;color:#f87171;border:1px solid #991b1b}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:#101a2e;border:1px solid #1e293b;border-radius:12px;padding:16px}
.card-head{display:flex;justify-content:space-between;align-items:center}
.card .name{font-weight:700;text-transform:capitalize}
.chip{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;
font-weight:600;background:#1e293b;color:#cbd5e1;margin-right:6px}
.chip.ok{background:#052e16;color:#4ade80}
.chip.warn{background:#422006;color:#fbbf24}
.chip.bad{background:#450a0a;color:#f87171}
.chip.na{background:#1e293b;color:#64748b}
.bar{height:8px;background:#1e293b;border-radius:999px;margin:12px 0 6px;overflow:hidden}
.fill{height:100%;border-radius:999px}
.fill.ok{background:#22c55e}.fill.warn{background:#f59e0b}.fill.bad{background:#ef4444}
.fill.na{background:#475569}
.score-line{font-size:13px;color:#cbd5e1}
.meta{margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:#101a2e;
border:1px solid #1e293b;border-radius:12px;overflow:hidden}
th{background:#16213a;text-align:left;padding:10px 12px;color:#94a3b8;
font-size:12px;text-transform:uppercase;letter-spacing:.6px}
td{padding:9px 12px;border-top:1px solid #1e293b}
footer{margin-top:40px;border-top:1px solid #1e293b;padding-top:16px;
color:#64748b;font-size:12.5px}
code{background:#1e293b;padding:1px 6px;border-radius:6px;font-size:12px}
"""


def build_html(result: dict[str, Any]) -> str:
    """Render the scoring result into a complete HTML document string."""
    verdict = result.get("verdict", "n/a")
    vcls = _VERDICT_CLASS.get(verdict, "na")
    ds = result.get("dataset", {})
    factors_html = _factor_cards(result.get("factors", {}))
    gates_html = _gates_table(result.get("gates", []))
    columns_html = _columns_table(result.get("columns", []))
    worst_html = _worst_rows_table(result.get("worst_rows", []))
    card_html = _model_card_html(result.get("model_card", {}))
    thresholds = result.get("thresholds", {})
    tool = result.get("tool", {})
    tp = _esc(thresholds.get("factor_pass"))
    tw = _esc(thresholds.get("factor_warn"))
    cp = _esc(thresholds.get("composite_pass"))
    cw = _esc(thresholds.get("composite_warn"))
    crit = _esc(", ".join(thresholds.get("critical_factors") or []))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DQFS report - {_esc(ds.get('name'))}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>Data Quality Factor Scoring</h1>
    <p class="muted">{_esc(ds.get('name'))} &middot; {_esc(ds.get('source'))}</p>
  </div>
  <span class="badge {vcls}">{_esc(verdict)}</span>
</header>

<section class="hero">
  {_gauge_svg(float(result.get('dqs', 0.0)), verdict)}
  <dl class="kv">
    <dt>Generated</dt><dd>{_esc(result.get('generated_at'))}</dd>
    <dt>Rows x Cols</dt><dd>{_esc(ds.get('rows'))} x {_esc(ds.get('columns'))}</dd>
    <dt>Target</dt><dd>{_esc(ds.get('target'))}</dd>
    <dt>Run ID</dt><dd>{_esc(result.get('run_id'))}</dd>
    <dt>Duration</dt><dd>{_esc(result.get('duration_seconds'))}s</dd>
  </dl>
</section>

<h2>Quality Factors</h2>
<div class="grid">
{factors_html}
</div>

<h2>Gates</h2>
{gates_html}

<h2>Columns</h2>
{columns_html}

<h2>Most Suspicious Rows (deep ensemble)</h2>
{worst_html}

<h2>Model Card</h2>
<div class="card">{card_html}</div>

<footer>
  <p>Thresholds: factor pass &ge; {tp}, warn &ge; {tw};
  composite pass &ge; {cp}, warn &ge; {cw}.</p>
  <p>Critical factors: {crit}</p>
  <p>Generated by {_esc(tool.get('name'))} v{_esc(tool.get('version'))} &middot;
  run <code>{_esc(result.get('run_id'))}</code></p>
</footer>
</div>
</body>
</html>"""


def write_html(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Write the standalone HTML report.

    Raises:
        ReportError: If rendering or writing fails.
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"report_{result.get('run_id', 'unknown')}.html"
        path.write_text(build_html(result), encoding="utf-8")
        logger.info("HTML report written: %s", path)
        return path
    except OSError as exc:
        raise ReportError(f"failed to write HTML report: {exc}") from exc
