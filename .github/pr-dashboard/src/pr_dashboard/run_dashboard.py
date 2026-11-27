import argparse
import dataclasses
import enum
import json
import pathlib

from pr_dashboard import board_mirror
from pr_dashboard import github_api
from pr_dashboard import load_labels
from pr_dashboard import pr_commands
from pr_dashboard import pr_context
from pr_dashboard import pr_labels

LABELS_TOML_PATH = pathlib.Path(__file__).parents[2] / "labels.toml"


class LabelUpdateActions(enum.StrEnum):
    CREATE = "create"
    UPDATE = "update"
    LEAVE = "leave"


##
## === ACTIONS
##


def print_update_plan(
    update_plan: pr_context.UpdatePlan,
) -> None:
    print(
        json.dumps(
            dataclasses.asdict(update_plan),
            indent=2,
        ),
    )


def label_pr(
    *,
    repo: str,
    pr_number: int,
    dry_run: bool,
) -> None:
    configured_labels = load_labels.load_configured_labels(LABELS_TOML_PATH)
    pull_request = github_api.get_pull_request(
        repo=repo,
        pr_number=pr_number,
    )
    update_plan_from_pr = pr_labels.plan_update_from_pr(
        configured_labels=configured_labels,
        pull_request=pull_request,
    )
    print_update_plan(update_plan_from_pr)
    if not dry_run:
        github_api.apply_update_plan(
            repo=repo,
            pr_number=pr_number,
            comment_id=None,
            update_plan=update_plan_from_pr,
        )


def run_command_from_comment(
    *,
    repo: str,
    event_path: pathlib.Path,
) -> None:
    with open(event_path) as event_file:
        event_data = json.load(event_file)
    if "pull_request" not in event_data["issue"]:
        return
    comment_text = event_data["comment"]["body"]
    if len(pr_commands.extract_mentioned_commands(comment_text)) == 0:
        return
    user_name = event_data["comment"]["user"]["login"]
    user = github_api.get_user(
        repo=repo,
        user_name=user_name,
    )
    comment = pr_context.Comment(
        comment_id=event_data["comment"]["id"],
        user=user,
        text=comment_text,
    )
    pr_number = event_data["issue"]["number"]
    pull_request = github_api.get_pull_request(
        repo=repo,
        pr_number=pr_number,
    )
    open_milestones = github_api.get_open_milestones(repo)
    closed_milestones = github_api.get_closed_milestones(repo)
    configured_labels = load_labels.load_configured_labels(LABELS_TOML_PATH)
    update_plan_from_comment = pr_commands.plan_update_from_comment(
        configured_labels=configured_labels,
        pull_request=pull_request,
        comment=comment,
        open_milestones=open_milestones,
        closed_milestones=closed_milestones,
    )
    print_update_plan(update_plan_from_comment)
    github_api.apply_update_plan(
        repo=repo,
        pr_number=pr_number,
        comment_id=comment.comment_id,
        update_plan=update_plan_from_comment,
    )


def print_board_change(
    board_change: board_mirror.BoardChange,
) -> None:
    change_parts = []
    if board_change.status_to_set is not None:
        change_parts.append(f"Status -> {board_change.status_to_set}")
    if board_change.review_by_date_changes:
        change_parts.append(f"Review by date -> {board_change.review_by_date_to_set}")
    print(f"#{board_change.pr_number}: {'; '.join(change_parts)}")


def mirror_to_board(
    *,
    repo: str,
    project_number: int,
    dry_run: bool,
) -> None:
    owner = repo.split("/")[0]
    board = github_api.get_board(
        owner=owner,
        project_number=project_number,
    )
    activities = github_api.get_open_pull_request_activities(
        repo=repo,
        project_number=project_number,
    )
    num_changes = 0
    for activity in activities:
        if activity.board_item is None:
            print(f"#{activity.pr_number}: not on the board, so skipped")
            continue
        board_change = board_mirror.plan_board_change(activity)
        if board_change is None:
            continue
        print_board_change(board_change)
        num_changes += 1
        if not dry_run:
            github_api.apply_board_change(
                board=board,
                board_change=board_change,
            )
    print(f"{num_changes} of {len(activities)} open PRs changed on the board")


def get_label_update_action(
    *,
    label: pr_labels.Label,
    repo_labels: dict[str, dict],
) -> LabelUpdateActions:
    existing_label_data = repo_labels.get(label.label_name)
    if existing_label_data is None:
        return LabelUpdateActions.CREATE
    elif ((existing_label_data["color"].lower() == label.color.lower())
          and (existing_label_data["description"] == label.description)):
        return LabelUpdateActions.LEAVE
    else:
        return LabelUpdateActions.UPDATE


def update_label_definitions(
    *,
    repo: str,
    dry_run: bool,
) -> None:
    configured_labels = load_labels.load_configured_labels(LABELS_TOML_PATH)
    repo_labels = github_api.get_repo_labels(repo)
    for label in configured_labels.all_labels():
        update_action = get_label_update_action(
            label=label,
            repo_labels=repo_labels,
        )
        if update_action == LabelUpdateActions.LEAVE:
            continue
        print(f"{update_action} {label.label_name}")
        if dry_run:
            continue
        elif update_action == LabelUpdateActions.CREATE:
            github_api.create_label(
                repo=repo,
                name=label.label_name,
                color=label.color,
                description=label.description,
            )
        elif update_action == LabelUpdateActions.UPDATE:
            github_api.update_label(
                repo=repo,
                name=label.label_name,
                color=label.color,
                description=label.description,
            )
        else:
            raise ValueError(f"`update_label_definitions` has no branch for action `{update_action}`.")


##
## === ARGUMENTS
##


def add_label_pr_parser(
    action_parsers: argparse._SubParsersAction,
) -> None:
    label_pr_parser = action_parsers.add_parser("label-pr")
    label_pr_parser.add_argument(
        "--repo",
        type=str,
        required=True,
    )
    label_pr_parser.add_argument(
        "--pr",
        type=int,
        required=True,
    )
    label_pr_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )


def add_run_command_parser(
    action_parsers: argparse._SubParsersAction,
) -> None:
    run_command_parser = action_parsers.add_parser("run-command")
    run_command_parser.add_argument(
        "--repo",
        type=str,
        required=True,
    )
    run_command_parser.add_argument(
        "--event-path",
        type=pathlib.Path,
        required=True,
    )


def add_update_label_definitions_parser(
    action_parsers: argparse._SubParsersAction,
) -> None:
    update_label_definitions_parser = action_parsers.add_parser("update-label-definitions")
    update_label_definitions_parser.add_argument(
        "--repo",
        type=str,
        required=True,
    )
    update_label_definitions_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )


def add_mirror_to_board_parser(
    action_parsers: argparse._SubParsersAction,
) -> None:
    mirror_to_board_parser = action_parsers.add_parser("mirror-to-board")
    mirror_to_board_parser.add_argument(
        "--repo",
        type=str,
        required=True,
    )
    mirror_to_board_parser.add_argument(
        "--project-number",
        type=int,
        required=True,
    )
    mirror_to_board_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    parser = argparse.ArgumentParser()
    action_parsers = parser.add_subparsers(
        dest="action",
        required=True,
    )
    add_label_pr_parser(action_parsers)
    add_run_command_parser(action_parsers)
    add_update_label_definitions_parser(action_parsers)
    add_mirror_to_board_parser(action_parsers)
    parsed_args = parser.parse_args()
    if parsed_args.action == "label-pr":
        label_pr(
            repo=parsed_args.repo,
            pr_number=parsed_args.pr,
            dry_run=parsed_args.dry_run,
        )
    elif parsed_args.action == "run-command":
        run_command_from_comment(
            repo=parsed_args.repo,
            event_path=parsed_args.event_path,
        )
    elif parsed_args.action == "update-label-definitions":
        update_label_definitions(
            repo=parsed_args.repo,
            dry_run=parsed_args.dry_run,
        )
    elif parsed_args.action == "mirror-to-board":
        mirror_to_board(
            repo=parsed_args.repo,
            project_number=parsed_args.project_number,
            dry_run=parsed_args.dry_run,
        )
    else:
        raise ValueError(f"`main` has no branch for action `{parsed_args.action}`.")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()
