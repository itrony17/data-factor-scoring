#!/usr/bin/env bash
# Create the GitHub repository and push DQFS in one command.
#
# Usage:
#   export GITHUB_TOKEN=ghp_xxx        # PAT with "repo" scope (fine-grained: Contents read/write)
#   bash scripts/create_github_repo.sh [owner/repo] [description]
#
# Defaults: owner taken from the token, repo name "data-factor-scoring".
# The token is read from the environment only - it is never written to disk
# or embedded in the remote URL (an askpass helper feeds it to git).
set -euo pipefail

REPO_NAME="${1:-data-factor-scoring}"
DESCRIPTION="${2:-DQFS - Data Quality Factor Scoring: deep-ML data quality gates with GitHub-native reporting}"

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN first: export GITHUB_TOKEN=...}"
: "${GITHUB_USER:=$(curl -fsS -H "Authorization: Bearer ${GITHUB_TOKEN}" https://api.github.com/user | sed -n 's/.*"login": *"\([^"]*\)".*/\1/p')}"
: "${GITHUB_USER:?Could not resolve the token owner - check the token}"

REPO_SLUG="${REPO_NAME}"
[[ "$REPO_SLUG" != */* ]] && REPO_SLUG="${GITHUB_USER}/${REPO_SLUG}"

echo "==> Creating ${REPO_SLUG} (private=false) ..."
HTTP_CODE=$(curl -sS -o /tmp/gh_repo_create.json -w '%{http_code}' \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"${REPO_SLUG#*/}\",\"description\":\"${DESCRIPTION}\",\"private\":false}")
if [[ "$HTTP_CODE" == "201" ]]; then
  echo "    created."
elif [[ "$HTTP_CODE" == "422" ]]; then
  echo "    already exists - reusing it."
else
  echo "    GitHub API returned HTTP ${HTTP_CODE}:" >&2
  cat /tmp/gh_repo_create.json >&2
  exit 1
fi

cd "$(dirname "$0")/.."

# Feed the token to git without putting it in the remote URL or argv.
ASKPASS="$(mktemp)"
cat >"$ASKPASS" <<'EOF'
#!/usr/bin/env sh
printf '%s\n' "$GITHUB_TOKEN"
EOF
chmod +x "$ASKPASS"
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  git remote add origin "https://github.com/${REPO_SLUG}.git" 2>/dev/null ||
  GIT_ASKPASS="$ASKPASS" git remote set-url origin "https://github.com/${REPO_SLUG}.git"

echo "==> Pushing main + tags ..."
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git push -u origin main
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git push --tags || true
rm -f "$ASKPASS"

echo
echo "==> Done: https://github.com/${REPO_SLUG}"
echo "    CI (ruff + pytest on 3.10-3.12) starts automatically on the first push."
echo "    Optional: enable GitHub Pages -> Settings > Pages > Deploy from branch > main /docs,"
echo "    then run: dqfs publish --repo ${REPO_SLUG} --pages --pages-branch main"
