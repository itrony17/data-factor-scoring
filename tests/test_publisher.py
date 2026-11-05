"""Tests for dqfs.github.publisher - fully mocked, no network access."""

from __future__ import annotations

import base64
import json
import os
import time
from unittest.mock import MagicMock

import pytest

from dqfs.errors import PublishError
from dqfs.github.publisher import (
    GitHubPublisher,
    collect_result_files,
    resolve_token,
)


@pytest.fixture()
def results_dir(tmp_path):
    """A fake results directory with one run's artifacts."""
    files = {
        "latest.json": b'{"dqs": 88.0}',
        "latest.csv": b"dqs\n88.0\n",
        "report_20260101_090000.html": b"<html>old</html>",
        "report_20260102_090000.html": b"<html>new</html>",
        "scores_20260102_090000.xlsx": b"\x00PK-binary",
    }
    t = time.time() - 600
    for i, (name, content) in enumerate(files.items()):
        path = tmp_path / name
        path.write_bytes(content)
        os.utime(path, (t + i * 60, t + i * 60))  # deterministic mtimes
    return tmp_path


def _scripted_publisher(responses: dict[tuple[str, str], object]) -> GitHubPublisher:
    """Publisher whose _request returns scripted values per (method, path)."""
    pub = GitHubPublisher(token="tok", repo="acme/quality")
    defaults = {
        ("GET", "/repos/acme/quality"): {"default_branch": "main"},
        ("GET", "/repos/acme/quality/git/ref/heads/gh-results"): {
            "object": {"sha": "basesha"}
        },
        ("GET", "/repos/acme/quality/git/commits/basesha"): {
            "tree": {"sha": "basetree"}
        },
        ("POST", "/repos/acme/quality/git/blobs"): {"sha": "blobsha"},
        ("POST", "/repos/acme/quality/git/trees"): {"sha": "newtree"},
        ("POST", "/repos/acme/quality/git/commits"): {"sha": "newsha"},
        ("PATCH", "/repos/acme/quality/git/refs/heads/gh-results"): {},
        ("POST", "/repos/acme/quality/statuses/newsha"): {"state": "success"},
    }

    def fake_request(method: str, path: str, **kwargs):
        key = (method, path.split("?")[0])
        if key in responses:
            return responses[key]
        if key in defaults:
            return defaults[key]
        raise AssertionError(f"unexpected API call {method} {path}")

    pub._request = MagicMock(side_effect=fake_request)
    return pub


def test_resolve_token_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "  abc123  ")
    assert resolve_token() == "abc123"


def test_resolve_token_cli_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "envtok")
    assert resolve_token("clitok") == "clitok"


def test_resolve_token_missing_raises(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(PublishError, match="GITHUB_TOKEN"):
        resolve_token()


def test_collect_result_files_picks_latest(results_dir) -> None:
    files = collect_result_files(results_dir)
    assert set(files) == {
        "latest.json", "latest.csv",
        "report_20260102_090000.html", "scores_20260102_090000.xlsx",
    }
    assert files["latest.json"] == b'{"dqs": 88.0}'


def test_collect_result_files_missing_dir(tmp_path) -> None:
    with pytest.raises(PublishError, match="not found"):
        collect_result_files(tmp_path / "nope")


def test_publish_commits_results_files(results_dir) -> None:
    pub = _scripted_publisher({})
    out = pub.publish(results_dir, branch="gh-results", verdict="PASS")
    assert out.commit_sha == "newsha"
    assert not out.dry_run
    paths = out.files
    assert "results/20260102_090000/latest.json" in paths
    assert "results/latest/latest.json" in paths
    assert "results/latest/report_20260102_090000.html" in paths
    # commit status posted for the verdict
    posted = [
        c for c in pub._request.call_args_list
        if c.args[:2] == ("POST", "/repos/acme/quality/statuses/newsha")
    ]
    assert posted and posted[0].kwargs["json"]["state"] == "success"


def test_publish_pages_updates_docs_index(results_dir) -> None:
    calls = []

    def scripted(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if method == "GET" and "/git/ref/heads/" in path:
            branch = path.rsplit("/", 1)[-1]
            return {"object": {"sha": f"head-{branch}"}}
        if method == "GET" and "git/commits/" in path:
            return {"tree": {"sha": "basetree"}}
        if method == "POST" and path.endswith("/git/refs"):
            return {}
        if method == "POST" and path.endswith("/git/blobs"):
            return {"sha": f"blob{len(calls)}"}
        if method == "POST" and path.endswith("/git/trees"):
            return {"sha": "newtree"}
        if method == "POST" and path.endswith("/git/commits"):
            return {"sha": f"commit{len(calls)}"}
        if method == "PATCH" and "/git/refs/heads/" in path:
            return {}
        return {"sha": "fallback"}

    pub = GitHubPublisher(token="tok", repo="acme/quality")
    pub._request = MagicMock(side_effect=scripted)
    out = pub.publish(results_dir, pages=True, pages_branch="main")
    assert out.pages_updated
    # the newest HTML dashboard was pushed as a plain utf-8 blob
    html_blobs = [
        c[2]["json"]["content"]
        for c in calls
        if c[0] == "POST" and c[1].endswith("/git/blobs")
        and c[2]["json"]["encoding"] == "utf-8"
    ]
    assert any(c.startswith("<html>new") for c in html_blobs)
    # pages commit landed on the pages branch (main)
    assert any(
        c[1] == "/repos/acme/quality/git/refs/heads/main" and c[0] == "PATCH"
        for c in calls
    )


def test_publish_dry_run_makes_no_calls(results_dir) -> None:
    pub = _scripted_publisher({})
    out = pub.publish(results_dir, dry_run=True)
    assert out.dry_run and out.commit_sha == "dry-run"
    pub._request.assert_not_called()


def test_publish_without_artifacts_raises(tmp_path) -> None:
    pub = _scripted_publisher({})
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PublishError, match="no publishable artifacts"):
        pub.publish(empty)


def test_auth_error_is_typed_with_hint(results_dir) -> None:
    pub = GitHubPublisher(token="tok", repo="acme/quality")
    response = MagicMock(status_code=401, json=lambda: {"message": "Bad credentials"})
    pub.session.request = MagicMock(return_value=response)  # HTTP layer mock
    with pytest.raises(PublishError, match="auth/permission"):
        pub.commit_files("gh-results", {"results/a.json": b"{}"}, "msg")


def test_xlsx_is_base64_encoded(results_dir) -> None:
    payloads = []

    def scripted(method, path, **kwargs):
        if method == "GET" and "/git/ref/heads/" in path:
            return {"object": {"sha": "head"}}
        if method == "GET" and "git/commits/" in path:
            return {"tree": {"sha": "basetree"}}
        if method == "POST" and path.endswith("/git/blobs"):
            payloads.append(kwargs["json"])
            return {"sha": f"b{len(payloads)}"}
        if method == "POST" and path.endswith("/git/trees"):
            return {"sha": "newtree"}
        if method == "POST" and path.endswith("/git/commits"):
            return {"sha": "newsha"}
        if method == "PATCH" and "/git/refs/heads/" in path:
            return {}
        return {}

    pub = GitHubPublisher(token="tok", repo="acme/quality")
    pub._request = MagicMock(side_effect=scripted)
    pub.publish(results_dir, branch="gh-results")
    xlsx_blobs = [p for p in payloads if p["encoding"] == "base64"]
    assert xlsx_blobs
    decoded = base64.b64decode(xlsx_blobs[0]["content"])
    assert decoded.startswith(b"\x00PK")
    json_blobs = [p for p in payloads if p["encoding"] == "utf-8"]
    assert json_blobs and json.loads(json_blobs[0]["content"])["dqs"] == 88.0
