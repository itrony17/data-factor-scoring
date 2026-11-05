"""JSON report writer: ``latest.json`` plus a timestamped archive copy."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dqfs.errors import ReportError

logger = logging.getLogger(__name__)


def write_json(result: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write the scoring result as JSON.

    Args:
        result: JSON-serializable scoring result (from
            :func:`dqfs.scoring.run_scoring`).
        out_dir: Destination directory (created if missing).

    Returns:
        List of written paths: ``latest.json`` and
        ``scores_<run_id>.json``.

    Raises:
        ReportError: If serialization or writing fails.
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        run_id = result.get("run_id", "unknown")
        paths = []
        for name in ("latest.json", f"scores_{run_id}.json"):
            path = out / name
            with path.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            paths.append(path)
        logger.info("JSON report written: %s", paths[0])
        return paths
    except (OSError, TypeError, ValueError) as exc:
        raise ReportError(f"failed to write JSON report: {exc}") from exc
