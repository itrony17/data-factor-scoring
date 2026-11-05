"""Excel and CSV report writers.

The Excel workbook contains five sheets (Summary, Factors, Gates, Columns,
Worst Rows) with tuned column widths; the CSV export is a flat factor table
for quick diffing and downstream tooling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from dqfs.errors import ReportError

logger = logging.getLogger(__name__)

_SHEET_WIDTHS = {
    "Summary": [26, 60],
    "Factors": [16, 10, 10, 10, 10, 16],
    "Gates": [22, 10, 10, 46],
    "Columns": [20, 12, 12, 12, 12, 16],
    "Worst Rows": [10, 10, 70],
}


def _summary_frame(result: dict[str, Any]) -> pd.DataFrame:
    ds = result.get("dataset", {})
    thresholds = result.get("thresholds", {})
    pairs = [
        ("run_id", result.get("run_id")),
        ("generated_at", result.get("generated_at")),
        ("tool", f"{result.get('tool', {}).get('name')} {result.get('tool', {}).get('version')}"),
        ("dataset", ds.get("name")),
        ("source", ds.get("source")),
        ("rows", ds.get("rows")),
        ("columns", ds.get("columns")),
        ("target", ds.get("target")),
        ("DQS", result.get("dqs")),
        ("verdict", result.get("verdict")),
        ("factor_pass_threshold", thresholds.get("factor_pass")),
        ("factor_warn_threshold", thresholds.get("factor_warn")),
        ("composite_pass", thresholds.get("composite_pass")),
        ("composite_warn", thresholds.get("composite_warn")),
        ("critical_factors", ", ".join(thresholds.get("critical_factors") or [])),
        ("duration_seconds", result.get("duration_seconds")),
    ]
    return pd.DataFrame(pairs, columns=["field", "value"])


def _factors_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, info in result.get("factors", {}).items():
        rows.append({
            "factor": name,
            "score": info.get("score"),
            "status": info.get("status"),
            "weight": info.get("weight"),
            "ic": info.get("ic"),
            "deep_importance": info.get("deep_importance"),
        })
    return pd.DataFrame(rows)


def write_excel(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Write the multi-sheet Excel workbook ``scores_<run_id>.xlsx``.

    Raises:
        ReportError: If workbook creation fails.
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"scores_{result.get('run_id', 'unknown')}.xlsx"
        frames = {
            "Summary": _summary_frame(result),
            "Factors": _factors_frame(result),
            "Gates": pd.DataFrame(result.get("gates", [])),
            "Columns": pd.DataFrame(result.get("columns", [])),
            "Worst Rows": pd.DataFrame(
                [
                    {
                        "row_index": w.get("row_index"),
                        "badness": w.get("badness"),
                        "contributions": str(w.get("contributions")),
                    }
                    for w in result.get("worst_rows", [])
                ]
            ),
        }
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet, index=False)
                worksheet = writer.sheets[sheet]
                widths = _SHEET_WIDTHS.get(sheet, [18] * max(len(frame.columns), 1))
                for i, width in enumerate(widths[: len(frame.columns)]):
                    worksheet.column_dimensions[
                        worksheet.cell(row=1, column=i + 1).column_letter
                    ].width = width
        logger.info("Excel report written: %s", path)
        return path
    except (OSError, ValueError) as exc:
        raise ReportError(f"failed to write Excel report: {exc}") from exc


def write_csv(result: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write the flat factor table as CSV.

    Returns:
        ``latest.csv`` and ``scores_<run_id>.csv`` paths.
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frame = _factors_frame(result)
        frame.insert(0, "dqs", result.get("dqs"))
        frame.insert(1, "verdict", result.get("verdict"))
        run_id = result.get("run_id", "unknown")
        paths = []
        for name in ("latest.csv", f"scores_{run_id}.csv"):
            path = out / name
            frame.to_csv(path, index=False)
            paths.append(path)
        logger.info("CSV report written: %s", paths[0])
        return paths
    except OSError as exc:
        raise ReportError(f"failed to write CSV report: {exc}") from exc
