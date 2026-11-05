"""Report generation subpackage.

Each scoring run produces a consistent set of artifacts:

* ``latest.json`` + ``scores_<run_id>.json``  - machine-readable results;
* ``report_<run_id>.html``                    - standalone HTML dashboard;
* ``scores_<run_id>.csv`` / ``latest.csv``    - flat factor table;
* ``scores_<run_id>.xlsx``                    - multi-sheet Excel workbook.
"""

from dqfs.report.excel_report import write_csv, write_excel
from dqfs.report.html_report import write_html
from dqfs.report.json_report import write_json

__all__ = ["write_csv", "write_excel", "write_html", "write_json"]
