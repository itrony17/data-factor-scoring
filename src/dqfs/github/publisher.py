"""Publish scoring results to a GitHub repository via the REST API.

Uses the GitHub Git data API (blobs -> tree -> commit -> ref) so a run can
commit a whole result set atomically to a reporting branch, optionally:

* post a commit **status** for the quality gate (context ``dqfs/quality-gate``);
* update a **GitHub Pages** source folder with the latest HTML dashboard.

Only the ``GITHUB_TOKEN`` environment variable (or ``--token``) is required;
see ``.env.example``. All HTTP access is through :mod:`requests` and is
fully mocked in the test suite - no network needed for tests.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from dqfs.errors import PublishError

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_RESULTS_PREFIX = "results"
_PAGES_PREFIX = "docs"
_PUBLISH_GLOBS = ("latest.json", "latest.csv", "report_*.html", "scores_*.xlsx")
_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.5

_STATUS_MAP = {"PASS": "success", "WARN": "failure", "FAIL": "failure"}


@dataclass
class PublishResult:
    """Summary of a publish operation."""

    repo: str
    branch: str
    commit_sha: str
    files: list[str] = field(default_factory=list)
    pages_updated: bool = False
    status_created: bool = False
    dry_run: bool = False


def resolve_token(cli_token: str | None = None) -> str:
    """Resolve the GitHub token from CLI arg or ``GITHUB_TOKEN`` env var.

    Raises:
        PublishError: If no token is available, with setup guidance.
    """
    token = cli_token or os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise PublishError(
            "GitHub token not found. Export GITHUB_TOKEN (see .env.example) "
            "or pass --token. A fine-grained token needs Contents: read/write."
        )
    return token


def collect_result_files(results_dir: str | Path) -> dict[str, bytes]:
    """Collect the latest publishable artifacts from a results directory.

    Returns:
        Mapping of file name -> bytes (``latest.json``, ``latest.csv``, the
        newest ``report_*.html`` and ``scores_*.xlsx``).
    """
    root = Path(results_dir)
    if not root.exists():
        raise PublishError(f"results directory not found: {root}")
    picked: dict[str, Path] = {}
    for pattern in _PUBLISH_GLOBS:
        matches = sorted(
            root.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name)
        )
        if matches:
            picked[matches[-1].name] = matches[-1]
    return {name: path.read_bytes() for name, path in picked.items()}


class GitHubPublisher:
    """Thin GitHub REST client focused on committing result artifacts."""

    def __init__(self, token: str, repo: str, timeout: float = 30.0) -> None:
        if "/" not in repo:
            raise PublishError(f"repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "dqfs-publisher",
        })

    # ------------------------------------------------------------- plumbing
    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Perform one API call with retry and typed error mapping."""
        url = f"{_API_BASE}{path}"
        last_error: str | None = None
        for attempt in range(_RETRIES + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                if attempt < _RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise PublishError(last_error) from exc
            if resp.status_code in (200, 201, 204):
                return resp.json() if resp.content else {}
            if 500 <= resp.status_code < 600 and attempt < _RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            message = _extract_message(resp)
            if resp.status_code in (401, 403):
                raise PublishError(
                    f"GitHub auth/permission error {resp.status_code}: {message}. "
                    "Check the token scope (Contents: read/write)."
                )
            if resp.status_code == 404:
                raise PublishError(
                    f"GitHub resource not found (404): {path}. "
                    f"Does repo {self.repo} exist and can the token see it?"
                )
            raise PublishError(f"GitHub API error {resp.status_code}: {message}")
        raise PublishError(f"GitHub API unreachable after retries: {last_error}")

    # ---------------------------------------------------------------- git api
    def default_branch(self) -> str:
        """Name of the repository's default branch."""
        data = self._request("GET", f"/repos/{self.repo}")
        return str(data.get("default_branch", "main"))

    def branch_sha(self, branch: str) -> str | None:
        """HEAD commit sha of ``branch`` or None when it does not exist."""
        try:
            data = self._request("GET", f"/repos/{self.repo}/git/ref/heads/{branch}")
        except PublishError as exc:
            if "404" in str(exc):
                return None
            raise
        return str(data["object"]["sha"])

    def _ensure_branch(self, branch: str) -> str:
        """Return the branch HEAD sha, creating the branch if needed."""
        sha = self.branch_sha(branch)
        if sha:
            return sha
        base = self.branch_sha(self.default_branch())
        if not base:
            raise PublishError(f"cannot resolve default branch of {self.repo}")
        self._request(
            "POST",
            f"/repos/{self.repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": base},
        )
        logger.info("created branch %s from %s", branch, base[:12])
        return base

    def commit_files(
        self, branch: str, files: dict[str, bytes], message: str
    ) -> str:
        """Atomically commit ``files`` (repo-relative paths) onto ``branch``.

        Returns:
            The new commit sha.
        """
        if not files:
            raise PublishError("no files to commit")
        base_sha = self._ensure_branch(branch)
        base_commit = self._request("GET", f"/repos/{self.repo}/git/commits/{base_sha}")
        base_tree = str(base_commit["tree"]["sha"])

        tree_entries = []
        for path, content in files.items():
            encoding = "base64" if b"\x00" in content or not _is_text(content) else "utf-8"
            payload = (
                base64.b64encode(content).decode("ascii")
                if encoding == "base64"
                else content.decode("utf-8")
            )
            blob = self._request(
                "POST",
                f"/repos/{self.repo}/git/blobs",
                json={"content": payload, "encoding": encoding},
            )
            tree_entries.append({
                "path": path, "mode": "100644", "type": "blob",
                "sha": blob["sha"],
            })
        tree = self._request(
            "POST",
            f"/repos/{self.repo}/git/trees",
            json={"base_tree": base_tree, "tree": tree_entries},
        )
        commit = self._request(
            "POST",
            f"/repos/{self.repo}/git/commits",
            json={"message": message, "tree": tree["sha"], "parents": [base_sha]},
        )
        new_sha = str(commit["sha"])
        self._request(
            "PATCH",
            f"/repos/{self.repo}/git/refs/heads/{branch}",
            json={"sha": new_sha, "force": False},
        )
        return new_sha

    def create_status(
        self, sha: str, verdict: str, target_url: str | None = None
    ) -> bool:
        """Post a commit status for the quality-gate verdict."""
        state = _STATUS_MAP.get(verdict, "error")
        self._request(
            "POST",
            f"/repos/{self.repo}/statuses/{sha}",
            json={
                "state": state,
                "context": "dqfs/quality-gate",
                "description": f"DQFS quality gate: {verdict}",
                "target_url": target_url,
            },
        )
        return True

    # --------------------------------------------------------------- publish
    def publish(
        self,
        results_dir: str | Path,
        branch: str = "gh-results",
        message: str | None = None,
        pages: bool = False,
        pages_branch: str | None = None,
        verdict: str | None = None,
        report_url: str | None = None,
        dry_run: bool = False,
    ) -> PublishResult:
        """Publish the latest result artifacts to ``branch``.

        Args:
            results_dir: Directory produced by a scoring run.
            branch: Target branch receiving ``results/`` artifacts.
            message: Commit message (defaults to a DQFS template).
            pages: Also push the newest HTML report to ``docs/index.html``
                on ``pages_branch`` for GitHub Pages.
            pages_branch: Branch Pages serves from (default: repo default).
            verdict: When given, post a commit status for the quality gate.
            report_url: Optional URL attached to the status.
            dry_run: Collect and validate the plan without any API calls.

        Returns:
            :class:`PublishResult` describing what was (or would be) done.
        """
        files = collect_result_files(results_dir)
        if not files:
            raise PublishError(
                f"no publishable artifacts found in {results_dir} "
                f"(looked for: {', '.join(_PUBLISH_GLOBS)})"
            )
        run_id = _run_id_from(files)
        repo_files = {
            f"{_RESULTS_PREFIX}/{run_id}/{name}": content for name, content in files.items()
        }
        repo_files.update({
            f"{_RESULTS_PREFIX}/latest/{name}": content for name, content in files.items()
        })
        commit_message = message or f"dqfs: scoring results {run_id}"
        if dry_run:
            return PublishResult(
                repo=self.repo, branch=branch, commit_sha="dry-run",
                files=sorted(repo_files), dry_run=True,
            )
        sha = self.commit_files(branch, repo_files, commit_message)
        logger.info(
            "published %d files to %s@%s (%s)", len(repo_files), self.repo, branch, sha[:12]
        )

        pages_updated = False
        if pages:
            html_name = next((n for n in files if n.startswith("report_")), None)
            if html_name:
                pages_target = {_PAGES_PREFIX + "/index.html": files[html_name]}
                target = pages_branch or self.default_branch()
                pages_sha = self.commit_files(
                    target, pages_target, f"dqfs: pages dashboard {run_id}"
                )
                pages_updated = True
                logger.info("pages updated on %s (%s)", target, pages_sha[:12])

        status_created = False
        if verdict:
            status_created = self.create_status(sha, verdict, report_url)
        return PublishResult(
            repo=self.repo, branch=branch, commit_sha=sha,
            files=sorted(repo_files), pages_updated=pages_updated,
            status_created=status_created,
        )


def _run_id_from(files: dict[str, bytes]) -> str:
    """Extract the run id from a ``report_<run_id>.html`` style name."""
    for name in files:
        if name.startswith("report_") and name.endswith(".html"):
            return name[len("report_"):-len(".html")]
    return "run"


def _is_text(content: bytes) -> bool:
    """Heuristic: true when content decodes as UTF-8 without NUL bytes."""
    try:
        content.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _extract_message(resp: requests.Response) -> str:
    """Best-effort extraction of GitHub's error message."""
    try:
        return str(resp.json().get("message", resp.text[:200]))
    except ValueError:
        return resp.text[:200]
