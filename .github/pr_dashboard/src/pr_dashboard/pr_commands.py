import dataclasses
import datetime
import enum
import re

from pr_dashboard import pr_context
from pr_dashboard import pr_labels

##
## === ACCESS RULES
##

_ROLES_WITH_TRIAGE_ACCESS = (
    pr_context.RoleInRepo.TRIAGE,
    pr_context.RoleInRepo.WRITE,
    pr_context.RoleInRepo.MAINTAIN,
    pr_context.RoleInRepo.ADMIN,
)
_ROLES_WITH_WRITE_ACCESS = (
    pr_context.RoleInRepo.WRITE,
    pr_context.RoleInRepo.MAINTAIN,
    pr_context.RoleInRepo.ADMIN,
)

##
## === COMMANDS
##


@dataclasses.dataclass(frozen=True)
class Command:
    text: str
    argument_hint: str
    description: str
    allowed_users: str

    @property
    def usage(
        self,
    ) -> str:
        if self.argument_hint == "":
            return f"`{self.text}`"
        else:
            return f"`{self.text} {self.argument_hint}`"


class Commands(enum.Enum):
    ACCEPT_REVIEW = Command(
        text="/accept-review",
        argument_hint="<YYYY-MM-DD>",
        description="take on this PR's review, with the date you will review by",
        allowed_users="maintainers (anyone with write access)",
    )
    RELEASE_REVIEW = Command(
        text="/release-review",
        argument_hint="",
        description="give the review back",
        allowed_users="the current reviewer",
    )
    SET_PRIORITY = Command(
        text="/set-priority",
        argument_hint="<{priority_values}>",
        description="set this PR's priority",
        allowed_users="the PR author, or maintainers",
    )
    CREATE_MILESTONE = Command(
        text="/create-milestone",
        argument_hint="<title>",
        description="create a new open milestone for a campaign",
        allowed_users="maintainers (anyone with write access)",
    )
    SET_MILESTONE = Command(
        text="/set-milestone",
        argument_hint="<title|none>",
        description="set or remove this PR's milestone",
        allowed_users="the PR author, or maintainers",
    )
    RECOMPUTE_LABELS = Command(
        text="/recompute-labels",
        argument_hint="",
        description="recalculate the scope and size labels",
        allowed_users="the PR author, or maintainers",
    )
    HELP = Command(
        text="/help",
        argument_hint="",
        description="show this list",
        allowed_users="anyone",
    )

    @property
    def text(
        self,
    ) -> str:
        return self.value.text

    @property
    def usage(
        self,
    ) -> str:
        return self.value.usage

    @property
    def description(
        self,
    ) -> str:
        return self.value.description

    @property
    def allowed_users(
        self,
    ) -> str:
        return self.value.allowed_users


##
## === PARSING
##


@dataclasses.dataclass(frozen=True)
class ParsedCommand:
    command: Commands
    argument: str


def parse_command(
    comment_text: str,
) -> ParsedCommand | None:
    comment_lines = comment_text.strip().splitlines()
    if len(comment_lines) == 0:
        return None
    first_line_words = comment_lines[0].split(maxsplit=1)
    matching_commands = [_command for _command in Commands if _command.text == first_line_words[0]]
    if len(matching_commands) == 0:
        return None
    if len(first_line_words) == 1:
        argument = ""
    else:
        argument = first_line_words[1].strip()
    return ParsedCommand(
        command=matching_commands[0],
        argument=argument,
    )


def extract_mentioned_commands(
    comment_text: str,
) -> tuple[Commands, ...]:
    commands_by_text = {_command.text: _command for _command in Commands}
    command_text_options = "|".join(re.escape(_command_text) for _command_text in commands_by_text)
    command_pattern = re.compile(rf"(?<![\w/-])({command_text_options})(?![\w-])")
    mentioned_commands = []
    for mentioned_text in command_pattern.findall(comment_text):
        mentioned_command = commands_by_text[mentioned_text]
        if mentioned_command not in mentioned_commands:
            mentioned_commands.append(mentioned_command)
    return tuple(mentioned_commands)


##
## === REPLIES
##


def _get_usage_placeholder_values(
    configured_labels: pr_labels.ConfiguredLabels,
) -> dict[str, str]:
    priority_values = "|".join(
        _priority_label.value for _priority_label in configured_labels.priority_labels.labels
    )
    return {
        "priority_values": priority_values,
    }


def _format_usage(
    *,
    command: Commands,
    configured_labels: pr_labels.ConfiguredLabels,
) -> str:
    placeholder_values = _get_usage_placeholder_values(configured_labels)
    return command.usage.format(**placeholder_values)


def _plan_command_usage_reply(
    command: Commands,
) -> pr_context.UpdatePlan:
    return pr_context.UpdatePlan(reply=f"Use {command.usage}.")


def _plan_command_help_reply(
    configured_labels: pr_labels.ConfiguredLabels,
) -> pr_context.UpdatePlan:
    help_lines = ["PR dashboard commands (each must start its comment):"]
    for command in Commands:
        usage = _format_usage(
            command=command,
            configured_labels=configured_labels,
        )
        help_line = f"- {usage}: {command.description}. For {command.allowed_users}."
        help_lines.append(help_line)
    return pr_context.UpdatePlan(reply="\n".join(help_lines))


def _format_user_mentions(
    user_names: tuple[str, ...],
) -> str:
    mentions = [f"@{_user_name}" for _user_name in user_names]
    if len(mentions) == 1:
        return mentions[0]
    else:
        return ", ".join(mentions[:-1]) + " and " + mentions[-1]


def _format_review_taken_text(
    assignee_user_names: tuple[str, ...],
) -> str:
    if len(assignee_user_names) == 1:
        verb = "has"
    else:
        verb = "have"
    return (
        f"{_format_user_mentions(assignee_user_names)} {verb} accepted this review. "
        "You're welcome to leave a review or comment, or ask them to hand it over."
    )


def _plan_command_access_refusal(
    command: Commands,
) -> pr_context.UpdatePlan:
    return pr_context.UpdatePlan(
        reply=f"Only the PR author and maintainers can use `{command.text}`.",
    )


def _plan_command_placement_reply(
    mentioned_commands: tuple[Commands, ...],
) -> pr_context.UpdatePlan:
    command_texts = " or ".join(f"`{_command.text}`" for _command in mentioned_commands)
    if len(mentioned_commands) == 1:
        instruction = "post it at the start of a new comment"
    else:
        instruction = "post each at the start of its own comment"
    return pr_context.UpdatePlan(
        reply=f"Commands only run when they start a comment. To run {command_texts}, {instruction}.",
    )


##
## === COMMAND LOGIC
##


def _accept_review(
    *,
    pull_request: pr_context.PullRequest,
    comment: pr_context.Comment,
    argument: str,
) -> pr_context.UpdatePlan:
    try:
        review_by_date = datetime.date.fromisoformat(argument)
    except ValueError:
        review_by_date = None
    today_in_utc = datetime.datetime.now(datetime.timezone.utc).date()
    # a day of slack: reviewers give their local date, which can be a day behind UTC
    earliest_review_by_date = today_in_utc - datetime.timedelta(days=1)
    if comment.user.role_in_repo not in _ROLES_WITH_WRITE_ACCESS:
        return pr_context.UpdatePlan(reply="Only maintainers (anyone with write access) can accept a review.")
    elif comment.user.user_name == pull_request.author_user_name:
        return pr_context.UpdatePlan(reply="You can't accept the review of your own PR.")
    elif review_by_date is None:
        # argument could not be parsed
        return _plan_command_usage_reply(Commands.ACCEPT_REVIEW)
    elif review_by_date < earliest_review_by_date:
        return pr_context.UpdatePlan(
            reply=f"The review date has to be today or later. Use {Commands.ACCEPT_REVIEW.usage}.",
        )
    elif pull_request.assignee_user_names == (comment.user.user_name, ):
        # already the reviewer: acknowledge the new date
        return pr_context.UpdatePlan(react=True)
    elif len(pull_request.assignee_user_names) > 0:
        # someone else has the review: refuse, naming who
        return pr_context.UpdatePlan(reply=_format_review_taken_text(pull_request.assignee_user_names))
    else:
        # nobody has the review: the commenter takes it
        return pr_context.UpdatePlan(
            assignee_user_names_to_add=(comment.user.user_name, ),
            react=True,
        )


def _release_review(
    *,
    pull_request: pr_context.PullRequest,
    comment: pr_context.Comment,
    argument: str,
) -> pr_context.UpdatePlan:
    if argument != "":
        # the command received an argument even though it does not take one
        return _plan_command_usage_reply(Commands.RELEASE_REVIEW)
    elif comment.user.user_name in pull_request.assignee_user_names:
        # the reviewer gives the review back: unassign them
        return pr_context.UpdatePlan(
            assignee_user_names_to_remove=(comment.user.user_name, ),
            react=True,
        )
    elif len(pull_request.assignee_user_names) > 0:
        return pr_context.UpdatePlan(
            reply=
            f"{_format_user_mentions(pull_request.assignee_user_names)} accepted this review, so only they can release it.",
        )
    else:
        return pr_context.UpdatePlan(reply="Nobody has accepted this review yet.")


def _set_priority(
    *,
    configured_labels: pr_labels.ConfiguredLabels,
    pull_request: pr_context.PullRequest,
    argument: str,
) -> pr_context.UpdatePlan:
    configured_priority_labels = configured_labels.priority_labels
    chosen_priority_label = configured_priority_labels.get_label(argument)
    if chosen_priority_label is None:
        usage = _format_usage(
            command=Commands.SET_PRIORITY,
            configured_labels=configured_labels,
        )
        return pr_context.UpdatePlan(reply=f"Use {usage}.")
    chosen_priority_label_name = chosen_priority_label.label_name
    label_names_to_remove = tuple(
        _label_name for _label_name in pull_request.label_names
        if configured_priority_labels.check_has_prefix(_label_name) and
        (_label_name != chosen_priority_label_name)
    )
    if chosen_priority_label_name in pull_request.label_names:
        label_names_to_add = ()
    else:
        label_names_to_add = (chosen_priority_label_name, )
    return pr_context.UpdatePlan(
        label_names_to_add=label_names_to_add,
        label_names_to_remove=label_names_to_remove,
        react=True,
    )


def _create_milestone(
    *,
    comment: pr_context.Comment,
    open_milestones: tuple[pr_context.Milestone, ...],
    closed_milestones: tuple[pr_context.Milestone, ...],
    argument: str,
) -> pr_context.UpdatePlan:
    matching_open_milestones = [
        _milestone for _milestone in open_milestones if _milestone.title.lower() == argument.lower()
    ]
    matching_closed_milestones = [
        _milestone for _milestone in closed_milestones if _milestone.title.lower() == argument.lower()
    ]
    if comment.user.role_in_repo not in _ROLES_WITH_WRITE_ACCESS:
        return pr_context.UpdatePlan(
            reply="Only maintainers (anyone with write access) can create a milestone.",
        )
    elif argument == "":
        # the command received no milestone title
        return _plan_command_usage_reply(Commands.CREATE_MILESTONE)
    elif len(matching_open_milestones) > 0:
        return pr_context.UpdatePlan(
            reply=f'A milestone called "{matching_open_milestones[0].title}" is already open.',
        )
    elif len(matching_closed_milestones) > 0:
        return pr_context.UpdatePlan(
            reply=(
                f'A closed milestone called "{matching_closed_milestones[0].title}" exists. '
                "Reopen it on GitHub instead."
            ),
        )
    else:
        # the title is new: create an open milestone with it
        return pr_context.UpdatePlan(
            milestone_title_to_create=argument,
            react=True,
        )


def _set_milestone(
    *,
    open_milestones: tuple[pr_context.Milestone, ...],
    argument: str,
) -> pr_context.UpdatePlan:
    matching_milestones = [
        _milestone for _milestone in open_milestones if _milestone.title.lower() == argument.lower()
    ]
    if argument == "":
        # the command received no milestone title
        return _plan_command_usage_reply(Commands.SET_MILESTONE)
    elif argument.lower() == "none":
        # remove the PR's milestone
        return pr_context.UpdatePlan(
            milestone_change=pr_context.MilestoneChange(milestone_number=None),
            react=True,
        )
    elif len(matching_milestones) > 0:
        # set the PR's milestone to the open milestone with that title
        milestone_change = pr_context.MilestoneChange(
            milestone_number=matching_milestones[0].milestone_number,
        )
        return pr_context.UpdatePlan(
            milestone_change=milestone_change,
            react=True,
        )
    elif len(open_milestones) == 0:
        return pr_context.UpdatePlan(
            reply=f'No open milestone is called "{argument}", and there are no open milestones.',
        )
    else:
        milestone_titles = ", ".join(f"`{_milestone.title}`" for _milestone in open_milestones)
        return pr_context.UpdatePlan(
            reply=f'No open milestone is called "{argument}". The open milestones are {milestone_titles}.',
        )


def _recompute_labels(
    *,
    configured_labels: pr_labels.ConfiguredLabels,
    pull_request: pr_context.PullRequest,
    argument: str,
) -> pr_context.UpdatePlan:
    if argument != "":
        return _plan_command_usage_reply(Commands.RECOMPUTE_LABELS)
    update_plan_from_pr = pr_labels.plan_update_from_pr(
        configured_labels=configured_labels,
        pull_request=pull_request,
    )
    return dataclasses.replace(update_plan_from_pr, react=True)


##
## === UPDATE PLAN
##


def plan_update_from_comment(
    *,
    configured_labels: pr_labels.ConfiguredLabels,
    pull_request: pr_context.PullRequest,
    comment: pr_context.Comment,
    open_milestones: tuple[pr_context.Milestone, ...],
    closed_milestones: tuple[pr_context.Milestone, ...],
) -> pr_context.UpdatePlan | None:
    parsed_command = parse_command(comment.text)
    mentioned_commands = extract_mentioned_commands(comment.text)
    if (parsed_command is None) and (len(mentioned_commands) == 0):
        return None
    elif parsed_command is None:
        return _plan_command_placement_reply(mentioned_commands)
    command = parsed_command.command
    argument = parsed_command.argument
    commenter_is_author_or_triager = (
        (comment.user.user_name == pull_request.author_user_name)
        or (comment.user.role_in_repo in _ROLES_WITH_TRIAGE_ACCESS)
    )
    if command == Commands.HELP:
        return _plan_command_help_reply(configured_labels)
    elif command == Commands.ACCEPT_REVIEW:
        return _accept_review(
            pull_request=pull_request,
            comment=comment,
            argument=argument,
        )
    elif command == Commands.RELEASE_REVIEW:
        return _release_review(
            pull_request=pull_request,
            comment=comment,
            argument=argument,
        )
    elif command == Commands.CREATE_MILESTONE:
        return _create_milestone(
            comment=comment,
            open_milestones=open_milestones,
            closed_milestones=closed_milestones,
            argument=argument,
        )
    elif not commenter_is_author_or_triager:
        return _plan_command_access_refusal(command)
    elif command == Commands.SET_PRIORITY:
        return _set_priority(
            configured_labels=configured_labels,
            pull_request=pull_request,
            argument=argument,
        )
    elif command == Commands.SET_MILESTONE:
        return _set_milestone(
            open_milestones=open_milestones,
            argument=argument,
        )
    elif command == Commands.RECOMPUTE_LABELS:
        return _recompute_labels(
            configured_labels=configured_labels,
            pull_request=pull_request,
            argument=argument,
        )
    else:
        raise ValueError(f"`plan_update_from_comment` has no branch for command `{command.text}`.")
