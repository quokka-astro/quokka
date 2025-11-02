#!/usr/bin/env python3

"""
Download and extract a GitHub Actions artifact.

Examples:
  GITHUB_TOKEN=<token> python3 scripts/download_gha_artifact.py \
    --repo quokka-astro/quokka \
    --workflow build-reproducible-artifacts.yml \
    --branch development \
    --artifact quokka-cuda-reproducible-build \
    --output ./artifacts
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com"


def load_token(env_key: str = "GITHUB_TOKEN") -> str:
    token = os.environ.get(env_key)
    if not token:
        instructions = f"""
Environment variable {env_key} is not set.

Create a Personal Access Token (PAT) and export it before running this script:

1. Open GitHub → Settings → Developer settings → Personal access tokens.
2. Choose “Fine-grained token → Generate new token” (or “Tokens (classic)” if needed).
3. Name the token, set an expiry, and select the repository (or “All repositories”).
4. Under Repository permissions, grant Actions: Read.
5. Generate the token, copy the value, and in your shell run:

   export {env_key}=<your-token-value>

Then re-run this script.
"""
        sys.exit(instructions.strip())
    return token


def request_json(url: str, token: str, params: Dict[str, str] | None = None) -> Dict:
    if params:
        query = "&".join(f"{quote(k)}={quote(v)}" for k, v in params.items())
        url = f"{url}?{query}"
    req = Request(url, headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:
        return json.load(resp)


def download_zip(url: str, token: str) -> bytes:
    req = Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(req) as resp:
        return resp.read()


def list_workflow_runs(repo: str, workflow: str, branch: str, token: str) -> Iterable[Dict]:
    url = f"{API_ROOT}/repos/{repo}/actions/workflows/{quote(workflow)}/runs"
    data = request_json(url, token, params={"status": "success", "branch": branch, "per_page": "20"})
    return data.get("workflow_runs", [])


def list_run_artifacts(repo: str, run_id: int, token: str) -> Iterable[Dict]:
    url = f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/artifacts"
    data = request_json(url, token, params={"per_page": "100"})
    return data.get("artifacts", [])


def safe_extract(zip_bytes: bytes, output_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            member_path = output_dir.joinpath(member.filename).resolve()
            if not str(member_path).startswith(str(output_dir.resolve())):
                raise RuntimeError(f"Refusing to extract outside target directory: {member.filename}")
        zf.extractall(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a GitHub Actions artifact.")
    parser.add_argument("--repo", required=True, help="Repository in the form owner/repo.")
    parser.add_argument("--workflow", required=True, help="Workflow filename, e.g. build-reproducible-artifacts.yml.")
    parser.add_argument("--artifact", required=True, help="Artifact name to download.")
    parser.add_argument("--branch", default="development", help="Branch to scan for completed workflow runs.")
    parser.add_argument("--run-id", type=int, help="Explicit workflow run id to use.")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory to write the artifact into.")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Environment variable that stores the PAT.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = load_token(args.token_env)

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id
    if run_id is None:
        runs = list_workflow_runs(args.repo, args.workflow, args.branch, token)
        if not runs:
            sys.exit(f"No successful runs found for workflow '{args.workflow}' on branch '{args.branch}'.")
        run_id = runs[0]["id"]
        print(f"Using workflow run id {run_id} ({runs[0]['display_title']})")

    artifacts = list_run_artifacts(args.repo, run_id, token)
    if not artifacts:
        sys.exit(f"No artifacts found for run id {run_id}.")

    match = next((a for a in artifacts if a["name"] == args.artifact), None)
    if match is None:
        available = ", ".join(a["name"] for a in artifacts)
        sys.exit(f"Artifact '{args.artifact}' not found for run id {run_id}. Available: {available}")

    try:
        zip_bytes = download_zip(match["archive_download_url"], token)
    except HTTPError as exc:
        sys.exit(f"Failed to download artifact: {exc}")

    target_dir = output_dir / args.artifact
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_extract(zip_bytes, target_dir)
    print(f"Artifact '{args.artifact}' extracted to {target_dir}")


if __name__ == "__main__":
    main()
