import datetime
import http
import json
import subprocess
import sys
import urllib.parse

from pr_dashboard import board_mirror
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


def _run_graphql(
    *,
    query: str,
    variables: dict,
) -> dict:
    cmd_parts = [
        "gh",
        "api",
        "graphql",
        "--input",
        "-",
    ]
    request_text = json.dumps({"query": query, "variables": variables})
    response_text = _run_gh_api(
        cmd_parts=cmd_parts,
        input_text=request_text,
    )
    response_data = json.loads(response_text)
    if "errors" in response_data:
        raise RuntimeError(f"GraphQL request failed: {response_data['errors']}")
    return response_data["data"]


def _parse_github_time(
    time_text: str,
) -> datetime.datetime:
    return datetime.datetime.fromisoformat(time_text)


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
    except (RuntimeError, ValueError, KeyError) as error:
        print(f"Could not read the permission of `{user_name}`, so treating them as read-only: {error!r}", file=sys.stderr)
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


_BOARD_QUERY = """
query($owner: String!, $project_number: Int!) {
  organization(login: $owner) {
    projectV2(number: $project_number) {
      id
      status_field: field(name: "Status") {
        ... on ProjectV2SingleSelectField { id options { id name } }
      }
      review_by_field: field(name: "Review by date") {
        ... on ProjectV2Field { id }
      }
    }
  }
}
"""


def get_board(
    *,
    owner: str,
    project_number: int,
) -> board_mirror.Board:
    response_data = _run_graphql(
        query=_BOARD_QUERY,
        variables={"owner": owner, "project_number": project_number},
    )
    project_data = response_data["organization"]["projectV2"]
    status_field_data = project_data["status_field"]
    review_by_field_data = project_data["review_by_field"]
    if (status_field_data is None) or (review_by_field_data is None):
        raise RuntimeError(f"project {project_number} needs a `Status` and a `Review by date` field.")
    return board_mirror.Board(
        project_id=project_data["id"],
        status_field_id=status_field_data["id"],
        status_option_ids={_option_data["name"]: _option_data["id"] for _option_data in status_field_data["options"]},
        review_by_field_id=review_by_field_data["id"],
    )


_OPEN_PULL_REQUESTS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 25, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        isDraft
        reviewDecision
        author { login }
        assignees(first: 10) { nodes { login } }
        reviews(last: 50) { nodes { author { __typename login } state submittedAt } }
        timelineItems(last: 100, itemTypes: [READY_FOR_REVIEW_EVENT, REVIEW_REQUESTED_EVENT]) {
          nodes {
            __typename
            ... on ReadyForReviewEvent { createdAt }
            ... on ReviewRequestedEvent { createdAt }
          }
        }
        comments(last: 100) { nodes { author { login } body createdAt } }
        projectItems(first: 10) {
          nodes {
            id
            project { number }
            status: fieldValueByName(name: "Status") { ... on ProjectV2ItemFieldSingleSelectValue { name } }
            review_by: fieldValueByName(name: "Review by date") { ... on ProjectV2ItemFieldDateValue { date } }
          }
        }
      }
    }
  }
}
"""

_FEEDBACK_REVIEW_STATES = ("COMMENTED", "CHANGES_REQUESTED", "APPROVED", "DISMISSED")


def _get_board_item(
    *,
    pr_data: dict,
    project_number: int,
) -> board_mirror.BoardItem | None:
    matching_items = [
        _item_data for _item_data in pr_data["projectItems"]["nodes"]
        if _item_data["project"]["number"] == project_number
    ]
    if len(matching_items) == 0:
        return None
    item_data = matching_items[0]
    status_data = item_data["status"] or {}
    review_by_data = item_data["review_by"] or {}
    review_by_text = review_by_data.get("date")
    if review_by_text is None:
        review_by_date = None
    else:
        review_by_date = datetime.date.fromisoformat(review_by_text)
    return board_mirror.BoardItem(
        item_id=item_data["id"],
        status_name=status_data.get("name"),
        review_by_date=review_by_date,
    )


def _get_pull_request_activity(
    *,
    pr_data: dict,
    project_number: int,
) -> board_mirror.PullRequestActivity:
    author_user_name = pr_data["author"]["login"]
    feedback_times = tuple(
        _parse_github_time(_review_data["submittedAt"]) for _review_data in pr_data["reviews"]["nodes"]
        if (_review_data["author"] is not None) and (_review_data["author"]["__typename"] == "User")
        and (_review_data["author"]["login"] != author_user_name) and (_review_data["state"] in _FEEDBACK_REVIEW_STATES)
    )
    timeline_items = pr_data["timelineItems"]["nodes"]
    review_request_times = tuple(
        _parse_github_time(_item_data["createdAt"]) for _item_data in timeline_items
        if _item_data["__typename"] == "ReviewRequestedEvent"
    )
    ready_for_review_times = tuple(
        _parse_github_time(_item_data["createdAt"]) for _item_data in timeline_items
        if _item_data["__typename"] == "ReadyForReviewEvent"
    )
    posted_comments = tuple(
        board_mirror.PostedComment(
            user_name=_comment_data["author"]["login"],
            text=_comment_data["body"],
            created_at=_parse_github_time(_comment_data["createdAt"]),
        ) for _comment_data in pr_data["comments"]["nodes"]
        if _comment_data["author"] is not None
    )
    board_item = _get_board_item(
        pr_data=pr_data,
        project_number=project_number,
    )
    return board_mirror.PullRequestActivity(
        pr_number=pr_data["number"],
        is_draft=pr_data["isDraft"],
        is_approved=(pr_data["reviewDecision"] == "APPROVED"),
        assignee_user_names=tuple(_assignee_data["login"] for _assignee_data in pr_data["assignees"]["nodes"]),
        feedback_times=feedback_times,
        review_request_times=review_request_times,
        ready_for_review_times=ready_for_review_times,
        posted_comments=posted_comments,
        board_item=board_item,
    )


def get_open_pull_request_activities(
    *,
    repo: str,
    project_number: int,
) -> tuple[board_mirror.PullRequestActivity, ...]:
    owner, name = repo.split("/")
    activities = []
    cursor = None
    while True:
        response_data = _run_graphql(
            query=_OPEN_PULL_REQUESTS_QUERY,
            variables={"owner": owner, "name": name, "cursor": cursor},
        )
        pull_requests_data = response_data["repository"]["pullRequests"]
        for pr_data in pull_requests_data["nodes"]:
            activity = _get_pull_request_activity(
                pr_data=pr_data,
                project_number=project_number,
            )
            activities.append(activity)
        if not pull_requests_data["pageInfo"]["hasNextPage"]:
            return tuple(activities)
        cursor = pull_requests_data["pageInfo"]["endCursor"]


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


_SET_SINGLE_SELECT_MUTATION = """
mutation($project_id: ID!, $item_id: ID!, $field_id: ID!, $option_id: String!) {
  updateProjectV2ItemFieldValue(
    input: {projectId: $project_id, itemId: $item_id, fieldId: $field_id, value: {singleSelectOptionId: $option_id}}
  ) { projectV2Item { id } }
}
"""

_SET_DATE_MUTATION = """
mutation($project_id: ID!, $item_id: ID!, $field_id: ID!, $date: Date!) {
  updateProjectV2ItemFieldValue(
    input: {projectId: $project_id, itemId: $item_id, fieldId: $field_id, value: {date: $date}}
  ) { projectV2Item { id } }
}
"""

_CLEAR_FIELD_MUTATION = """
mutation($project_id: ID!, $item_id: ID!, $field_id: ID!) {
  clearProjectV2ItemFieldValue(input: {projectId: $project_id, itemId: $item_id, fieldId: $field_id}) {
    projectV2Item { id }
  }
}
"""


def apply_board_change(
    *,
    board: board_mirror.Board,
    board_change: board_mirror.BoardChange,
) -> None:
    item_variables = {
        "project_id": board.project_id,
        "item_id": board_change.item_id,
    }
    if board_change.status_to_set is not None:
        option_id = board.status_option_ids.get(board_change.status_to_set)
        if option_id is None:
            raise RuntimeError(f"the board has no Status option named `{board_change.status_to_set}`.")
        _run_graphql(
            query=_SET_SINGLE_SELECT_MUTATION,
            variables={**item_variables, "field_id": board.status_field_id, "option_id": option_id},
        )
    if board_change.review_by_date_changes and (board_change.review_by_date_to_set is None):
        _run_graphql(
            query=_CLEAR_FIELD_MUTATION,
            variables={**item_variables, "field_id": board.review_by_field_id},
        )
    elif board_change.review_by_date_changes:
        date_text = board_change.review_by_date_to_set.isoformat()
        _run_graphql(
            query=_SET_DATE_MUTATION,
            variables={**item_variables, "field_id": board.review_by_field_id, "date": date_text},
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
