import http
import json
import subprocess
import urllib.parse

from pr_dashboard import pr_context

##
## === REQUESTS
##

_WRITE_METHODS = (
    http.HTTPMethod.POST,
    http.HTTPMethod.PATCH,
    http.HTTPMethod.PUT,
    http.HTTPMethod.DELETE,
)


def _run_gh_api(
    *,
    cmd_parts: list[str],
    input_text: str | None,
) -> str:
    completed_process = subprocess.run(
        cmd_parts,
        input=input_text,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        cmd_text = " ".join(cmd_parts)
        error_text = completed_process.stderr.strip()
        raise RuntimeError(f"`{cmd_text}` failed: {error_text}")
    return completed_process.stdout


def _read_from_github(
    *,
    endpoint: str,
    paginate: bool = False,
) -> dict | list:
    cmd_parts = [
        "gh",
        "api",
        "--method",
        http.HTTPMethod.GET,
        endpoint,
    ]
    if paginate:
        cmd_parts.append("--paginate")
        cmd_parts.append("--slurp")
    response_text = _run_gh_api(
        cmd_parts=cmd_parts,
        input_text=None,
    )
    return json.loads(response_text)


def _write_to_github(
    *,
    endpoint: str,
    method: http.HTTPMethod,
    fields: dict | None = None,
) -> None:
    if method not in _WRITE_METHODS:
        write_method_names = ", ".join(f"`{_write_method}`" for _write_method in _WRITE_METHODS)
        raise ValueError(f"`method` must be one of {{{write_method_names}}}; got `{method}`.")
    cmd_parts = [
        "gh",
        "api",
        "--method",
        method,
        endpoint,
    ]
    if fields is None:
        input_text = None
    else:
        cmd_parts.extend(["--input", "-"])
        input_text = json.dumps(fields)
    _run_gh_api(
        cmd_parts=cmd_parts,
        input_text=input_text,
    )


def _quote(
    path_part: str,
) -> str:
    return urllib.parse.quote(path_part, safe="")


##
## === READS
##


def get_pull_request(
    *,
    repo: str,
    pr_number: int,
) -> pr_context.PullRequest:
    pr_data = _read_from_github(endpoint=f"repos/{repo}/pulls/{pr_number}")
    file_pages = _read_from_github(
        endpoint=f"repos/{repo}/pulls/{pr_number}/files?per_page=100",
        paginate=True,
    )
    edited_file_paths = tuple(
        _file_data["filename"] for _file_page in file_pages for _file_data in _file_page
    )
    return pr_context.PullRequest(
        pr_number=pr_number,
        author_user_name=pr_data["user"]["login"],
        assignee_user_names=tuple(_assignee_data["login"] for _assignee_data in pr_data["assignees"]),
        label_names=tuple(_label_data["name"] for _label_data in pr_data["labels"]),
        num_lines_changed=pr_data["additions"] + pr_data["deletions"],
        edited_file_paths=edited_file_paths,
    )


def get_user(
    *,
    repo: str,
    user_name: str,
) -> pr_context.User:
    try:
        permission_data = _read_from_github(
            endpoint=f"repos/{repo}/collaborators/{_quote(user_name)}/permission",
        )
        role_in_repo = pr_context.RoleInRepo(permission_data["role_name"])
    except (RuntimeError, ValueError, KeyError):
        role_in_repo = pr_context.RoleInRepo.READ
    return pr_context.User(
        user_name=user_name,
        role_in_repo=role_in_repo,
    )


def _get_milestones(
    *,
    repo: str,
    state: str,
) -> tuple[pr_context.Milestone, ...]:
    milestone_pages = _read_from_github(
        endpoint=f"repos/{repo}/milestones?state={state}&per_page=100",
        paginate=True,
    )
    return tuple(
        pr_context.Milestone(
            milestone_number=_milestone_data["number"],
            title=_milestone_data["title"],
        ) for _milestone_page in milestone_pages for _milestone_data in _milestone_page
    )


def get_open_milestones(
    repo: str,
) -> tuple[pr_context.Milestone, ...]:
    return _get_milestones(
        repo=repo,
        state="open",
    )


def get_closed_milestones(
    repo: str,
) -> tuple[pr_context.Milestone, ...]:
    return _get_milestones(
        repo=repo,
        state="closed",
    )


def get_repo_labels(
    repo: str,
) -> dict[str, dict]:
    label_pages = _read_from_github(
        endpoint=f"repos/{repo}/labels?per_page=100",
        paginate=True,
    )
    return {_label_data["name"]: _label_data for _label_page in label_pages for _label_data in _label_page}


##
## === WRITES
##


def create_label(
    *,
    repo: str,
    name: str,
    color: str,
    description: str,
) -> None:
    label_fields = {
        "name": name,
        "color": color,
        "description": description,
    }
    _write_to_github(
        endpoint=f"repos/{repo}/labels",
        method=http.HTTPMethod.POST,
        fields=label_fields,
    )


def update_label(
    *,
    repo: str,
    name: str,
    color: str,
    description: str,
) -> None:
    label_fields = {
        "color": color,
        "description": description,
    }
    _write_to_github(
        endpoint=f"repos/{repo}/labels/{_quote(name)}",
        method=http.HTTPMethod.PATCH,
        fields=label_fields,
    )


def apply_update_plan(
    *,
    repo: str,
    pr_number: int,
    comment_id: int | None,
    update_plan: pr_context.UpdatePlan,
) -> None:
    # every PR is also an issue: its labels, assignees, milestone and comments live under the issues API
    issue_endpoint = f"repos/{repo}/issues/{pr_number}"
    for label_name in update_plan.label_names_to_remove:
        _write_to_github(
            endpoint=f"{issue_endpoint}/labels/{_quote(label_name)}",
            method=http.HTTPMethod.DELETE,
        )
    if len(update_plan.label_names_to_add) > 0:
        _write_to_github(
            endpoint=f"{issue_endpoint}/labels",
            method=http.HTTPMethod.POST,
            fields={"labels": list(update_plan.label_names_to_add)},
        )
    if len(update_plan.assignee_user_names_to_remove) > 0:
        assignee_fields = {"assignees": list(update_plan.assignee_user_names_to_remove)}
        _write_to_github(
            endpoint=f"{issue_endpoint}/assignees",
            method=http.HTTPMethod.DELETE,
            fields=assignee_fields,
        )
    if len(update_plan.assignee_user_names_to_add) > 0:
        assignee_fields = {"assignees": list(update_plan.assignee_user_names_to_add)}
        _write_to_github(
            endpoint=f"{issue_endpoint}/assignees",
            method=http.HTTPMethod.POST,
            fields=assignee_fields,
        )
    if update_plan.milestone_change is not None:
        milestone_fields = {"milestone": update_plan.milestone_change.milestone_number}
        _write_to_github(
            endpoint=issue_endpoint,
            method=http.HTTPMethod.PATCH,
            fields=milestone_fields,
        )
    if update_plan.milestone_title_to_create is not None:
        _write_to_github(
            endpoint=f"repos/{repo}/milestones",
            method=http.HTTPMethod.POST,
            fields={"title": update_plan.milestone_title_to_create},
        )
    if update_plan.reply is not None:
        _write_to_github(
            endpoint=f"{issue_endpoint}/comments",
            method=http.HTTPMethod.POST,
            fields={"body": update_plan.reply},
        )
    if update_plan.react and (comment_id is not None):
        _write_to_github(
            endpoint=f"repos/{repo}/issues/comments/{comment_id}/reactions",
            method=http.HTTPMethod.POST,
            fields={"content": "+1"},
        )
