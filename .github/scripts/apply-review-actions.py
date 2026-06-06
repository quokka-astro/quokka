#!/usr/bin/env python3
"""Apply a validated subset of automated PR review-management actions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


AUTOMATIC_PREFIX = "[Automatic Post]:"
MAX_ACTIONS_PER_RUN = 20
MAX_REVIEWERS_PER_ACTION = 3
MAX_COMMENTS_PER_PR_PER_RUN = 1
RECENT_COMMENT_SECONDS = 7 * 24 * 60 * 60

ALLOWED_ACTION_TYPES = {"comment", "request_review"}
ALLOWED_REASONS = {
    "waiting_for_review",
    "stale_author",
    "no_reviewers",
    "assigned_reviewer",
}
ALLOWED_TEMPLATES = {
    "waiting_for_review",
    "stale_author",
    "assigned_reviewer",
}

SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|auth|bearer|credential|password|private[_-]?key|secret|token)", re.IGNORECASE)
SECRET_VALUE_PATTERNS = {
    "private key block": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9_]{36,}|github_pat_[A-Za-z0-9_]{80,})\b"),
    "OpenAI token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    "Anthropic token": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{32,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}


@dataclass(frozen=True)
class Action:
    type: str
    pr: int
    reason: str
    reviewers: tuple[str, ...]
    body_template: str | None = None


class SecretScanError(ValueError):
    pass


class GitHubClient:
    def __init__(self, repository: str, token: str):
        self.repository = repository
        self.base_url = f"https://api.github.com/repos/{repository}"
        self.token = token

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._request(f"{self.base_url}{path}", method, payload)

    def graphql(self, query: str, variables: dict[str, Any]) -> Any:
        response = self._request("https://api.github.com/graphql", "POST", {"query": query, "variables": variables})
        errors = response.get("errors")
        if errors:
            raise RuntimeError(f"GitHub GraphQL query failed: {errors}")
        return response.get("data")

    def _request(self, url: str, method: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "quokka-review-action-applier",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read()
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {url} failed: {error.code} {details}") from error
        if not data:
            return None
        return json.loads(data.decode("utf-8"))

    def paged(self, path: str) -> list[Any]:
        items: list[Any] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 11):
            chunk = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            if not isinstance(chunk, list):
                raise RuntimeError(f"Expected list response for {path}")
            items.extend(chunk)
            if len(chunk) < 100:
                break
        return items

    def check_runs(self, sha: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for page in range(1, 11):
            chunk = self.request("GET", f"/commits/{sha}/check-runs?per_page=100&page={page}")
            runs = chunk.get("check_runs")
            if not isinstance(runs, list):
                raise RuntimeError("Expected check_runs list response")
            items.extend(runs)
            if len(runs) < 100:
                break
        return items


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def clean_login(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("reviewer login must be a string")
    login = value.strip()
    if not login or login.startswith("@") or any(char.isspace() for char in login):
        raise ValueError(f"invalid reviewer login: {value!r}")
    return login


def scan_for_secrets(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SecretScanError(f"{path} contains a non-string object key")
            child_path = f"{path}.{key}"
            if SENSITIVE_KEY_RE.search(key):
                raise SecretScanError(f"{child_path} uses a sensitive field name")
            scan_for_secrets(child, child_path)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            scan_for_secrets(child, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return

    for label, pattern in SECRET_VALUE_PATTERNS.items():
        if pattern.search(value):
            raise SecretScanError(f"{path} contains a value matching {label}")


def load_actions(path: str) -> list[Action]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    scan_for_secrets(data)

    if not isinstance(data, dict):
        raise ValueError("action plan must be a JSON object")
    if data.get("version") != 1:
        raise ValueError("action plan version must be 1")
    raw_actions = data.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("action plan actions must be a list")
    if len(raw_actions) > MAX_ACTIONS_PER_RUN:
        raise ValueError(f"too many actions: {len(raw_actions)} > {MAX_ACTIONS_PER_RUN}")

    actions: list[Action] = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            raise ValueError(f"action {index} must be an object")
        unknown_keys = set(raw_action) - {"type", "pr", "reason", "reviewers", "body_template"}
        if unknown_keys:
            raise ValueError(f"action {index} has unsupported keys: {sorted(unknown_keys)}")

        action_type = raw_action.get("type")
        reason = raw_action.get("reason")
        pr = raw_action.get("pr")
        body_template = raw_action.get("body_template")
        reviewers_value = raw_action.get("reviewers", [])

        if action_type not in ALLOWED_ACTION_TYPES:
            raise ValueError(f"action {index} has unsupported type: {action_type!r}")
        if reason not in ALLOWED_REASONS:
            raise ValueError(f"action {index} has unsupported reason: {reason!r}")
        if not isinstance(pr, int) or pr <= 0:
            raise ValueError(f"action {index} has invalid pr: {pr!r}")
        if not isinstance(reviewers_value, list):
            raise ValueError(f"action {index} reviewers must be a list")
        reviewers = tuple(dict.fromkeys(clean_login(reviewer) for reviewer in reviewers_value))
        if len(reviewers) > MAX_REVIEWERS_PER_ACTION:
            raise ValueError(f"action {index} has too many reviewers")

        if action_type == "comment":
            if body_template not in ALLOWED_TEMPLATES:
                raise ValueError(f"action {index} has unsupported body_template: {body_template!r}")
            if reason != body_template:
                raise ValueError(f"action {index} reason must match body_template")
            if body_template in {"waiting_for_review", "assigned_reviewer"} and not reviewers:
                raise ValueError(f"action {index} template requires at least one reviewer")
        elif body_template is not None:
            raise ValueError(f"action {index} request_review must not include body_template")
        if action_type == "request_review" and not reviewers:
            raise ValueError(f"action {index} request_review requires reviewers")

        actions.append(
            Action(
                type=action_type,
                pr=pr,
                reason=reason,
                reviewers=reviewers,
                body_template=body_template,
            )
        )
    return actions


def render_comment(action: Action, pr: dict[str, Any]) -> str:
    author = pr["user"]["login"]
    reviewers = ", ".join(f"@{reviewer}" for reviewer in action.reviewers)
    if action.body_template == "waiting_for_review":
        return (
            "[Automatic Post]: This PR seems to be currently waiting for review.\n"
            f"{reviewers}, could you please take a look when you have a chance?"
        )
    if action.body_template == "stale_author":
        return (
            "[Automatic Post]: It has been a while since there was any activity on this PR.\n"
            f"@{author}, are you still working on it? If so, please go ahead, request review, "
            "close it, or request that someone else follow up."
        )
    if action.body_template == "assigned_reviewer":
        if len(action.reviewers) != 1:
            raise ValueError("assigned_reviewer comments require exactly one reviewer")
        return (
            f"[Automatic Post]: I have assigned @{action.reviewers[0]} as a reviewer based on git blame information.\n"
            "Thanks in advance for the help!"
        )
    raise ValueError(f"unsupported comment template: {action.body_template}")


def has_recent_automatic_comment(comments: list[dict[str, Any]], body: str) -> bool:
    first_line = body.splitlines()[0]
    now = utc_now()
    for comment in comments:
        comment_body = comment.get("body", "")
        if not isinstance(comment_body, str):
            continue
        if comment_body == body or comment_body.startswith(first_line):
            return True
        if comment_body.startswith(AUTOMATIC_PREFIX):
            created_at = comment.get("created_at")
            if isinstance(created_at, str) and (now - parse_time(created_at)).total_seconds() < RECENT_COMMENT_SECONDS:
                return True
    return False


def has_blocking_review(client: GitHubClient, pr_number: int) -> bool:
    reviews = client.paged(f"/pulls/{pr_number}/reviews")
    latest_by_user: dict[str, str] = {}
    for review in reviews:
        user = review.get("user") or {}
        login = user.get("login")
        state = review.get("state")
        if isinstance(login, str) and isinstance(state, str):
            latest_by_user[login] = state
    return any(state == "CHANGES_REQUESTED" for state in latest_by_user.values())


def has_open_review_threads(client: GitHubClient, pr_number: int) -> bool:
    owner, name = client.repository.split("/", 1)
    query = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          isResolved
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""
    cursor: str | None = None
    for _ in range(10):
        data = client.graphql(query, {"owner": owner, "name": name, "number": pr_number, "cursor": cursor})
        repository = data.get("repository") if isinstance(data, dict) else None
        pull_request = repository.get("pullRequest") if isinstance(repository, dict) else None
        review_threads = pull_request.get("reviewThreads") if isinstance(pull_request, dict) else None
        if not isinstance(review_threads, dict):
            raise RuntimeError(f"Expected reviewThreads response for PR #{pr_number}")
        nodes = review_threads.get("nodes")
        if not isinstance(nodes, list):
            raise RuntimeError(f"Expected reviewThreads nodes for PR #{pr_number}")
        for thread in nodes:
            if not isinstance(thread, dict) or not isinstance(thread.get("isResolved"), bool):
                raise RuntimeError(f"Expected reviewThreads isResolved value for PR #{pr_number}")
            if not thread["isResolved"]:
                return True
        page_info = review_threads.get("pageInfo")
        if not isinstance(page_info, dict) or not page_info.get("hasNextPage"):
            return False
        cursor = page_info.get("endCursor")
        if not isinstance(cursor, str):
            raise RuntimeError(f"Expected reviewThreads endCursor for PR #{pr_number}")
    raise RuntimeError(f"Too many review thread pages for PR #{pr_number}")


def checks_are_clean(client: GitHubClient, sha: str) -> bool:
    status = client.request("GET", f"/commits/{sha}/status")
    status_count = status.get("total_count", 0)
    if status_count and status.get("state") != "success":
        return False

    runs = client.check_runs(sha)
    if not runs:
        return status_count == 0 or status.get("state") == "success"
    allowed = {"success", "skipped", "neutral"}
    return all(run.get("status") == "completed" and run.get("conclusion") in allowed for run in runs)


def pr_is_clean(client: GitHubClient, pr: dict[str, Any]) -> bool:
    if pr.get("draft"):
        return False
    if pr.get("mergeable") is not True:
        time.sleep(2)
        pr = client.request("GET", f"/pulls/{pr['number']}")
    if pr.get("mergeable") is not True:
        return False
    sha = pr["head"]["sha"]
    return checks_are_clean(client, sha)


def collaborator_can_review(client: GitHubClient, reviewer: str) -> bool:
    escaped = quote(reviewer, safe="")
    try:
        permission = client.request("GET", f"/collaborators/{escaped}/permission")
    except RuntimeError as error:
        print(f"Skipping reviewer @{reviewer}: collaborator permission check failed: {error}", file=sys.stderr)
        return False
    return permission.get("permission") in {"read", "triage", "write", "maintain", "admin"}


def validate_reviewers(client: GitHubClient, pr: dict[str, Any], reviewers: tuple[str, ...], *, allow_requested: bool) -> list[str]:
    author = pr["user"]["login"]
    requested = {reviewer["login"] for reviewer in pr.get("requested_reviewers", [])}
    valid: list[str] = []
    for reviewer in reviewers:
        if reviewer == author:
            print(f"Skipping @{reviewer} on PR #{pr['number']}: reviewer is PR author")
            continue
        if reviewer in requested and not allow_requested:
            print(f"Skipping @{reviewer} on PR #{pr['number']}: review already requested")
            continue
        if collaborator_can_review(client, reviewer):
            valid.append(reviewer)
        else:
            print(f"Skipping @{reviewer} on PR #{pr['number']}: not a reviewer-capable collaborator")
    return valid


def apply_action(client: GitHubClient, action: Action, comments_by_pr: dict[int, int]) -> None:
    pr = client.request("GET", f"/pulls/{action.pr}")
    if pr.get("state") != "open":
        print(f"Skipping PR #{action.pr}: not open")
        return

    if action.reason in {"waiting_for_review", "no_reviewers", "assigned_reviewer"} and not pr_is_clean(client, pr):
        print(f"Skipping PR #{action.pr}: PR is not in a clean non-draft state")
        return

    if action.reason == "waiting_for_review":
        if has_blocking_review(client, action.pr) or has_open_review_threads(client, action.pr):
            print(f"Skipping PR #{action.pr}: review is not currently clear")
            return
    if action.reason == "no_reviewers" and (pr.get("requested_reviewers") or pr.get("requested_teams")):
        print(f"Skipping PR #{action.pr}: reviewers already requested")
        return

    reviewers = validate_reviewers(client, pr, action.reviewers, allow_requested=action.type == "comment")
    action = Action(action.type, action.pr, action.reason, tuple(reviewers), action.body_template)

    if action.type == "request_review":
        if not action.reviewers:
            print(f"Skipping PR #{action.pr}: no valid reviewers remain")
            return
        client.request("POST", f"/pulls/{action.pr}/requested_reviewers", {"reviewers": list(action.reviewers)})
        print(f"Requested reviewers on PR #{action.pr}: {', '.join('@' + reviewer for reviewer in action.reviewers)}")
        return

    if comments_by_pr.get(action.pr, 0) >= MAX_COMMENTS_PER_PR_PER_RUN:
        print(f"Skipping PR #{action.pr}: comment limit reached for this run")
        return
    body = render_comment(action, pr)
    comments = client.paged(f"/issues/{action.pr}/comments")
    if has_recent_automatic_comment(comments, body):
        print(f"Skipping PR #{action.pr}: matching or recent automatic comment already exists")
        return
    client.request("POST", f"/issues/{action.pr}/comments", {"body": body})
    comments_by_pr[action.pr] = comments_by_pr.get(action.pr, 0) + 1
    print(f"Commented on PR #{action.pr}: {action.reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Validate the action plan without applying actions")
    parser.add_argument("action_plan", help="Path to review-actions.json")
    args = parser.parse_args()

    actions = load_actions(args.action_plan)
    if args.validate_only:
        print(f"Validated {len(actions)} review action(s)")
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    if not repository:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2

    client = GitHubClient(repository, token)
    comments_by_pr: dict[int, int] = {}

    for action in actions:
        apply_action(client, action, comments_by_pr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
