#!/usr/bin/env bash
# Review the current branch's committed changes relative to local development.
# Quickstart: https://github.com/alibaba/open-code-review/blob/main/pages/src/content/docs/en/quickstart.md

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: scripts/bash/open-code-review.sh [ocr review options]

Run from the Git working tree to review the current branch against local
development, starting at their merge base. Uncommitted changes are excluded.
Update development yourself before running if you need a newer baseline.

One-time setup (Git >= 2.41 and Node.js >= 18 required):
    npm install -g @alibaba-group/open-code-review
    ocr config provider
    ocr llm test

Examples:
    scripts/bash/open-code-review.sh
    scripts/bash/open-code-review.sh --preview
    scripts/bash/open-code-review.sh --format json --audience agent

Additional options are passed to ocr review; do not pass --commit or override
--repo, --from, or --to. See ocr review --help for available options.
EOF
    exit 0
fi

if ! command -v ocr >/dev/null 2>&1; then
    echo "Error: ocr is missing. Install with: npm install -g @alibaba-group/open-code-review" >&2
    echo "Then configure it with: ocr config provider" >&2
    exit 1
fi

repo_root=$(git rev-parse --show-toplevel)
if ! branch=$(git symbolic-ref --quiet HEAD); then
    echo "Error: detached HEAD; check out the branch you want to review." >&2
    exit 1
fi

if ! git rev-parse --verify --quiet 'refs/heads/development^{commit}' >/dev/null; then
    echo "Error: local development branch is missing; create it before running this script." >&2
    exit 1
fi

exec ocr review "$@" --repo "$repo_root" --from refs/heads/development --to "$branch"
