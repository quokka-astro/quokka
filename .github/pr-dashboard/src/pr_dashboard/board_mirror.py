import dataclasses
import datetime
import enum

from pr_dashboard import pr_commands

## review requests this soon after a PR is marked ready are GitHub requesting its code owners, not the author asking again
_CODE_OWNER_REQUEST_WINDOW = datetime.timedelta(seconds=60)

##
## === BOARD
##


class Statuses(enum.StrEnum):
    DRAFT = "Draft"
    REQUESTED_REVIEWERS = "Requested reviewers"
    WITH_REVIEWER = "Under review: with reviewer"
    AUTHOR_ADDRESSING_FEEDBACK = "Under review: author is addressing feedback"
    WAITING_ON_REREVIEW = "Under review: waiting on re-review"
    APPROVED = "Approved"


@dataclasses.dataclass(frozen=True)
class Board:
    project_id: str
    status_field_id: str
    status_option_ids: dict[str, str]
    review_by_field_id: str


@dataclasses.dataclass(frozen=True)
class BoardItem:
    item_id: str
    status_name: str | None
    review_by_date: datetime.date | None


##
## === PULL REQUEST ACTIVITY
##


@dataclasses.dataclass(frozen=True)
class PostedComment:
    user_name: str
    text: str
    created_at: datetime.datetime


@dataclasses.dataclass(frozen=True)
class PullRequestActivity:
    pr_number: int
    is_draft: bool
    is_approved: bool
    assignee_user_names: tuple[str, ...]
    ## reviews by people other than the author; bot reviews are left out
    feedback_times: tuple[datetime.datetime, ...]
    review_request_times: tuple[datetime.datetime, ...]
    ready_for_review_times: tuple[datetime.datetime, ...]
    posted_comments: tuple[PostedComment, ...]
    board_item: BoardItem | None


##
## === STATUS
##


def _check_is_code_owner_request(
    *,
    request_time: datetime.datetime,
    ready_for_review_times: tuple[datetime.datetime, ...],
) -> bool:
    return any(
        (_ready_time <= request_time) and (request_time - _ready_time <= _CODE_OWNER_REQUEST_WINDOW)
        for _ready_time in ready_for_review_times
    )


def _check_is_rerequested(
    activity: PullRequestActivity,
) -> bool:
    latest_feedback_time = max(activity.feedback_times)
    return any(
        (_request_time > latest_feedback_time) and not _check_is_code_owner_request(
            request_time=_request_time,
            ready_for_review_times=activity.ready_for_review_times,
        ) for _request_time in activity.review_request_times
    )


def compute_status(
    activity: PullRequestActivity,
) -> Statuses:
    feedback_is_given = (len(activity.feedback_times) > 0)
    if activity.is_draft:
        return Statuses.DRAFT
    elif activity.is_approved:
        return Statuses.APPROVED
    elif feedback_is_given and _check_is_rerequested(activity):
        return Statuses.WAITING_ON_REREVIEW
    elif feedback_is_given:
        return Statuses.AUTHOR_ADDRESSING_FEEDBACK
    elif len(activity.assignee_user_names) > 0:
        return Statuses.WITH_REVIEWER
    else:
        return Statuses.REQUESTED_REVIEWERS


def find_review_by_date(
    activity: PullRequestActivity,
) -> datetime.date | None:
    reviewer_comments = [
        _comment for _comment in activity.posted_comments
        if _comment.user_name in activity.assignee_user_names
    ]
    newest_first_comments = sorted(reviewer_comments, key=lambda _comment: _comment.created_at, reverse=True)
    for comment in newest_first_comments:
        parsed_command = pr_commands.parse_command(comment.text)
        if (parsed_command is None) or (parsed_command.command != pr_commands.Commands.ACCEPT_REVIEW):
            continue
        try:
            review_by_date = datetime.date.fromisoformat(parsed_command.argument)
        except ValueError:
            continue
        earliest_review_by_date = comment.created_at.date() - datetime.timedelta(days=1)
        if review_by_date >= earliest_review_by_date:
            return review_by_date
    return None


##
## === BOARD CHANGES
##


@dataclasses.dataclass(frozen=True)
class BoardChange:
    pr_number: int
    item_id: str
    status_to_set: Statuses | None
    review_by_date_changes: bool
    review_by_date_to_set: datetime.date | None


def plan_board_change(
    activity: PullRequestActivity,
) -> BoardChange | None:
    if activity.board_item is None:
        return None
    status = compute_status(activity)
    review_by_date = find_review_by_date(activity)
    status_changes = (status != activity.board_item.status_name)
    review_by_date_changes = (review_by_date != activity.board_item.review_by_date)
    if (not status_changes) and (not review_by_date_changes):
        return None
    if status_changes:
        status_to_set = status
    else:
        status_to_set = None
    return BoardChange(
        pr_number=activity.pr_number,
        item_id=activity.board_item.item_id,
        status_to_set=status_to_set,
        review_by_date_changes=review_by_date_changes,
        review_by_date_to_set=review_by_date,
    )
