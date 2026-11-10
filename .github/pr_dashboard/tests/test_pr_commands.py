import datetime
import pathlib
import unittest

from pr_dashboard import load_labels
from pr_dashboard import pr_commands
from pr_dashboard import pr_context

CONFIGURED_LABELS = load_labels.load_configured_labels(pathlib.Path(__file__).parents[1] / "labels.toml")
TODAY_IN_UTC = datetime.datetime.now(datetime.timezone.utc).date()
TODAY = TODAY_IN_UTC.isoformat()
YESTERDAY = (TODAY_IN_UTC - datetime.timedelta(days=1)).isoformat()
TWO_DAYS_AGO = (TODAY_IN_UTC - datetime.timedelta(days=2)).isoformat()
NEXT_WEEK = (TODAY_IN_UTC + datetime.timedelta(days=7)).isoformat()
IN_TWO_WEEKS = (TODAY_IN_UTC + datetime.timedelta(days=14)).isoformat()
OPEN_MILESTONES = (
    pr_context.Milestone(
        milestone_number=6,
        title="Rewrite RHD module",
    ),
    pr_context.Milestone(
        milestone_number=8,
        title="photochem",
    ),
)
CLOSED_MILESTONES = (pr_context.Milestone(
    milestone_number=2,
    title="Microphysics integration",
), )


def make_pull_request(
    *,
    assignee_user_names: tuple[str, ...] = (),
    label_names: tuple[str, ...] = (),
    edited_file_paths: tuple[str, ...] = ("src/hydro/hydro_system.cpp", ),
    num_lines_changed: int = 12,
) -> pr_context.PullRequest:
    return pr_context.PullRequest(
        pr_number=1,
        author_user_name="pr-author",
        assignee_user_names=assignee_user_names,
        label_names=label_names,
        num_lines_changed=num_lines_changed,
        edited_file_paths=edited_file_paths,
    )


DEFAULT_PULL_REQUEST = make_pull_request()


def run_command(
    *,
    text: str,
    pull_request: pr_context.PullRequest = DEFAULT_PULL_REQUEST,
    user_name: str = "maintainer",
    role_in_repo: pr_context.RoleInRepo = pr_context.RoleInRepo.WRITE,
    open_milestones: tuple[pr_context.Milestone, ...] = OPEN_MILESTONES,
    closed_milestones: tuple[pr_context.Milestone, ...] = CLOSED_MILESTONES,
) -> pr_context.UpdatePlan | None:
    user = pr_context.User(
        user_name=user_name,
        role_in_repo=role_in_repo,
    )
    comment = pr_context.Comment(
        comment_id=99,
        user=user,
        text=text,
    )
    return pr_commands.plan_update_from_comment(
        configured_labels=CONFIGURED_LABELS,
        pull_request=pull_request,
        comment=comment,
        open_milestones=open_milestones,
        closed_milestones=closed_milestones,
    )


class TestPrCommands_Parse(unittest.TestCase):

    def test_plain_comment_is_not_a_command(
        self,
    ):
        self.assertIsNone(pr_commands.parse_command("Looks good, thanks!"))

    def test_misspelt_command_is_ignored(
        self,
    ):
        self.assertIsNone(pr_commands.parse_command("/accept-reveiw 2026-11-12"))

    def test_command_mid_comment_is_ignored(
        self,
    ):
        self.assertIsNone(pr_commands.parse_command("I will run /release-review later"))

    def test_command_with_argument_and_extra_lines(
        self,
    ):
        parsed_command = pr_commands.parse_command(
            "/set-milestone Rewrite RHD module\nThis belongs to the rewrite.",
        )
        expected_parsed_command = pr_commands.ParsedCommand(
            command=pr_commands.Commands.SET_MILESTONE,
            argument="Rewrite RHD module",
        )
        self.assertEqual(parsed_command, expected_parsed_command)

    def test_command_without_argument(
        self,
    ):
        parsed_command = pr_commands.parse_command("/release-review")
        expected_parsed_command = pr_commands.ParsedCommand(
            command=pr_commands.Commands.RELEASE_REVIEW,
            argument="",
        )
        self.assertEqual(parsed_command, expected_parsed_command)

    def test_empty_comment(
        self,
    ):
        self.assertIsNone(pr_commands.parse_command("   "))


class TestPrCommands_Placement(unittest.TestCase):

    def test_command_mid_sentence_gets_a_hint(
        self,
    ):
        update_plan_from_comment = run_command(text="Thanks! /set-priority high")
        expected_reply = (
            "Commands only run when they start a comment. "
            "To run `/set-priority`, post it at the start of a new comment."
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_command_on_a_later_line_gets_a_hint(
        self,
    ):
        update_plan_from_comment = run_command(text="Looks good.\n/release-review")
        self.assertTrue(
            update_plan_from_comment.reply.startswith("Commands only run when they start a comment."),
        )

    def test_several_commands_are_each_named(
        self,
    ):
        update_plan_from_comment = run_command(text="Use /help or /set-priority low")
        expected_reply = (
            "Commands only run when they start a comment. "
            "To run `/help` or `/set-priority`, post each at the start of its own comment."
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_command_inside_a_url_path_is_not_a_command(
        self,
    ):
        self.assertIsNone(run_command(text="See https://example.org/docs/help for more."))

    def test_misspelt_command_gets_no_hint(
        self,
    ):
        self.assertIsNone(run_command(text="Thanks! /accept-reveiw tomorrow"))

    def test_comment_without_commands_gets_nothing(
        self,
    ):
        self.assertIsNone(run_command(text="Looks good, thanks!"))


class TestPrCommands_AcceptReview(unittest.TestCase):

    def test_free_review_is_assigned(
        self,
    ):
        update_plan_from_comment = run_command(text=f"/accept-review {NEXT_WEEK}")
        expected_update_plan = pr_context.UpdatePlan(
            assignee_user_names_to_add=("maintainer", ),
            react=True,
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_today_is_allowed(
        self,
    ):
        update_plan_from_comment = run_command(text=f"/accept-review {TODAY}")
        self.assertEqual(update_plan_from_comment.assignee_user_names_to_add, ("maintainer", ))

    def test_yesterday_is_allowed_for_reviewers_behind_utc(
        self,
    ):
        update_plan_from_comment = run_command(text=f"/accept-review {YESTERDAY}")
        self.assertEqual(update_plan_from_comment.assignee_user_names_to_add, ("maintainer", ))

    def test_same_reviewer_moving_the_date(
        self,
    ):
        pull_request = make_pull_request(assignee_user_names=("maintainer", ))
        update_plan_from_comment = run_command(
            text=f"/accept-review {IN_TWO_WEEKS}",
            pull_request=pull_request,
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(react=True))

    def test_second_reviewer_is_refused(
        self,
    ):
        pull_request = make_pull_request(assignee_user_names=("first-reviewer", ))
        update_plan_from_comment = run_command(
            text=f"/accept-review {NEXT_WEEK}",
            pull_request=pull_request,
        )
        expected_reply = (
            "@first-reviewer has accepted this review. "
            "You're welcome to leave a review or comment, or ask them to hand it over."
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_several_assignees_set_by_hand(
        self,
    ):
        pull_request = make_pull_request(assignee_user_names=("reviewer-a", "reviewer-b"))
        update_plan_from_comment = run_command(
            text=f"/accept-review {NEXT_WEEK}",
            pull_request=pull_request,
        )
        self.assertTrue(
            update_plan_from_comment.reply.
            startswith("@reviewer-a and @reviewer-b have accepted this review."),
        )

    def test_missing_date(
        self,
    ):
        update_plan_from_comment = run_command(text="/accept-review")
        self.assertEqual(update_plan_from_comment.reply, f"Use {pr_commands.Commands.ACCEPT_REVIEW.usage}.")

    def test_unreadable_date(
        self,
    ):
        update_plan_from_comment = run_command(text="/accept-review next week")
        self.assertEqual(update_plan_from_comment.reply, f"Use {pr_commands.Commands.ACCEPT_REVIEW.usage}.")

    def test_past_date(
        self,
    ):
        update_plan_from_comment = run_command(text=f"/accept-review {TWO_DAYS_AGO}")
        self.assertTrue(
            update_plan_from_comment.reply.startswith("The review date has to be today or later."),
        )

    def test_without_write_access(
        self,
    ):
        update_plan_from_comment = run_command(
            text=f"/accept-review {NEXT_WEEK}",
            user_name="contributor",
            role_in_repo=pr_context.RoleInRepo.READ,
        )
        expected_update_plan = pr_context.UpdatePlan(
            reply="Only maintainers (anyone with write access) can accept a review.",
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_triage_is_not_enough(
        self,
    ):
        update_plan_from_comment = run_command(
            text=f"/accept-review {NEXT_WEEK}",
            user_name="triager",
            role_in_repo=pr_context.RoleInRepo.TRIAGE,
        )
        expected_update_plan = pr_context.UpdatePlan(
            reply="Only maintainers (anyone with write access) can accept a review.",
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_author_cannot_accept_own_pr(
        self,
    ):
        update_plan_from_comment = run_command(
            text=f"/accept-review {NEXT_WEEK}",
            user_name="pr-author",
        )
        expected_update_plan = pr_context.UpdatePlan(reply="You can't accept the review of your own PR.")
        self.assertEqual(update_plan_from_comment, expected_update_plan)


class TestPrCommands_ReleaseReview(unittest.TestCase):

    def test_reviewer_releases(
        self,
    ):
        pull_request = make_pull_request(assignee_user_names=("maintainer", ))
        update_plan_from_comment = run_command(
            text="/release-review",
            pull_request=pull_request,
        )
        expected_update_plan = pr_context.UpdatePlan(
            assignee_user_names_to_remove=("maintainer", ),
            react=True,
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_someone_else_cannot_release(
        self,
    ):
        pull_request = make_pull_request(assignee_user_names=("first-reviewer", ))
        update_plan_from_comment = run_command(
            text="/release-review",
            pull_request=pull_request,
        )
        expected_reply = "@first-reviewer accepted this review, so only they can release it."
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_nobody_has_accepted(
        self,
    ):
        update_plan_from_comment = run_command(text="/release-review")
        self.assertEqual(
            update_plan_from_comment,
            pr_context.UpdatePlan(reply="Nobody has accepted this review yet."),
        )

    def test_unexpected_argument(
        self,
    ):
        pull_request = make_pull_request(assignee_user_names=("maintainer", ))
        update_plan_from_comment = run_command(
            text="/release-review now",
            pull_request=pull_request,
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply="Use `/release-review`."))


class TestPrCommands_SetPriority(unittest.TestCase):

    def test_replaces_existing_priority(
        self,
    ):
        pull_request = make_pull_request(label_names=("priority/normal", "scope/hydro"))
        update_plan_from_comment = run_command(
            text="/set-priority High",
            pull_request=pull_request,
        )
        expected_update_plan = pr_context.UpdatePlan(
            label_names_to_add=("priority/high", ),
            label_names_to_remove=("priority/normal", ),
            react=True,
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_same_priority_changes_nothing(
        self,
    ):
        pull_request = make_pull_request(label_names=("priority/low", ))
        update_plan_from_comment = run_command(
            text="/set-priority low",
            pull_request=pull_request,
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(react=True))

    def test_unknown_priority(
        self,
    ):
        update_plan_from_comment = run_command(text="/set-priority urgent")
        self.assertEqual(
            update_plan_from_comment,
            pr_context.UpdatePlan(reply="Use `/set-priority <low|normal|high>`."),
        )

    def test_author_with_read_access_can_set(
        self,
    ):
        update_plan_from_comment = run_command(
            text="/set-priority high",
            user_name="pr-author",
            role_in_repo=pr_context.RoleInRepo.READ,
        )
        self.assertEqual(update_plan_from_comment.label_names_to_add, ("priority/high", ))

    def test_triager_can_set(
        self,
    ):
        update_plan_from_comment = run_command(
            text="/set-priority high",
            user_name="triager",
            role_in_repo=pr_context.RoleInRepo.TRIAGE,
        )
        self.assertEqual(update_plan_from_comment.label_names_to_add, ("priority/high", ))

    def test_outsider_is_refused(
        self,
    ):
        update_plan_from_comment = run_command(
            text="/set-priority high",
            user_name="passer-by",
            role_in_repo=pr_context.RoleInRepo.READ,
        )
        expected_reply = "Only the PR author and maintainers can use `/set-priority`."
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))


class TestPrCommands_CreateMilestone(unittest.TestCase):

    def test_new_title_is_created(
        self,
    ):
        update_plan_from_comment = run_command(text="/create-milestone stellar-evolution")
        expected_update_plan = pr_context.UpdatePlan(
            milestone_title_to_create="stellar-evolution",
            react=True,
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_missing_title(
        self,
    ):
        update_plan_from_comment = run_command(text="/create-milestone")
        self.assertEqual(
            update_plan_from_comment,
            pr_context.UpdatePlan(reply="Use `/create-milestone <title>`."),
        )

    def test_open_title_is_refused_ignoring_case(
        self,
    ):
        update_plan_from_comment = run_command(text="/create-milestone Photochem")
        expected_reply = 'A milestone called "photochem" is already open.'
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_closed_title_is_refused(
        self,
    ):
        update_plan_from_comment = run_command(text="/create-milestone Microphysics integration")
        expected_reply = (
            'A closed milestone called "Microphysics integration" exists. '
            "Reopen it on GitHub instead."
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_without_write_access(
        self,
    ):
        update_plan_from_comment = run_command(
            text="/create-milestone stellar-evolution",
            user_name="pr-author",
            role_in_repo=pr_context.RoleInRepo.READ,
        )
        expected_update_plan = pr_context.UpdatePlan(
            reply="Only maintainers (anyone with write access) can create a milestone.",
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)


class TestPrCommands_SetMilestone(unittest.TestCase):

    def test_matches_title_ignoring_case(
        self,
    ):
        update_plan_from_comment = run_command(text="/set-milestone rewrite rhd module")
        expected_update_plan = pr_context.UpdatePlan(
            milestone_change=pr_context.MilestoneChange(milestone_number=6),
            react=True,
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_none_removes(
        self,
    ):
        update_plan_from_comment = run_command(text="/set-milestone none")
        expected_update_plan = pr_context.UpdatePlan(
            milestone_change=pr_context.MilestoneChange(milestone_number=None),
            react=True,
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_no_match_lists_open_milestones(
        self,
    ):
        update_plan_from_comment = run_command(text="/set-milestone RHD rewrite")
        expected_reply = (
            'No open milestone is called "RHD rewrite". '
            "The open milestones are `Rewrite RHD module`, `photochem`."
        )
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_no_open_milestones(
        self,
    ):
        update_plan_from_comment = run_command(
            text="/set-milestone photochem",
            open_milestones=(),
        )
        expected_reply = 'No open milestone is called "photochem", and there are no open milestones.'
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply=expected_reply))

    def test_missing_title(
        self,
    ):
        update_plan_from_comment = run_command(text="/set-milestone")
        self.assertEqual(
            update_plan_from_comment,
            pr_context.UpdatePlan(reply="Use `/set-milestone <title|none>`."),
        )


class TestPrCommands_RecomputeLabels(unittest.TestCase):

    def test_recomputes_scope_and_size(
        self,
    ):
        pull_request = make_pull_request(
            label_names=("scope/IO", "priority/high"),
            num_lines_changed=250,
        )
        update_plan_from_comment = run_command(
            text="/recompute-labels",
            pull_request=pull_request,
        )
        expected_update_plan = pr_context.UpdatePlan(
            label_names_to_add=("scope/hydro", "size/L"),
            label_names_to_remove=("scope/IO", ),
            react=True,
        )
        self.assertEqual(update_plan_from_comment, expected_update_plan)

    def test_unexpected_argument(
        self,
    ):
        update_plan_from_comment = run_command(text="/recompute-labels all")
        self.assertEqual(update_plan_from_comment, pr_context.UpdatePlan(reply="Use `/recompute-labels`."))


class TestPrCommands_Help(unittest.TestCase):

    def test_lists_every_command(
        self,
    ):
        update_plan_from_comment = run_command(text="/help")
        for command in pr_commands.Commands:
            with self.subTest(command_text=command.text):
                self.assertIn(f"`{command.text}", update_plan_from_comment.reply)

    def test_fills_in_the_priority_values(
        self,
    ):
        update_plan_from_comment = run_command(text="/help")
        self.assertIn(
            "`/set-priority <low|normal|high>`: set this PR's priority",
            update_plan_from_comment.reply,
        )
        self.assertNotIn("{priority_values}", update_plan_from_comment.reply)

    def test_anyone_can_ask(
        self,
    ):
        update_plan_from_comment = run_command(
            text="/help",
            user_name="passer-by",
            role_in_repo=pr_context.RoleInRepo.READ,
        )
        self.assertTrue(
            update_plan_from_comment.reply.startswith("PR dashboard commands (each must start its comment):"),
        )

    def test_extra_words_are_ignored(
        self,
    ):
        update_plan_from_comment = run_command(text="/help me please")
        self.assertTrue(
            update_plan_from_comment.reply.startswith("PR dashboard commands (each must start its comment):"),
        )

    def test_changes_nothing(
        self,
    ):
        update_plan_from_comment = run_command(text="/help")
        self.assertEqual(
            update_plan_from_comment,
            pr_context.UpdatePlan(reply=update_plan_from_comment.reply),
        )


if __name__ == "__main__":
    unittest.main()
