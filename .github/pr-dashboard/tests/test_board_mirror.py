import dataclasses
import datetime
import unittest

from pr_dashboard import board_mirror

START_TIME = datetime.datetime(2026, 11, 1, 12, 0, tzinfo=datetime.timezone.utc)
BOARD_ITEM = board_mirror.BoardItem(
    item_id="item",
    status_name="Requested reviewers",
    review_by_date=None,
)


def hours_after_start(
    num_hours: float,
) -> datetime.datetime:
    return START_TIME + datetime.timedelta(hours=num_hours)


def make_activity(
    **changed_facts,
) -> board_mirror.PullRequestActivity:
    activity = board_mirror.PullRequestActivity(
        pr_number=1,
        is_draft=False,
        is_approved=False,
        assignee_user_names=(),
        feedback_times=(),
        review_request_times=(hours_after_start(0), ),
        ready_for_review_times=(),
        posted_comments=(),
        board_item=BOARD_ITEM,
    )
    return dataclasses.replace(activity, **changed_facts)


def make_comment(
    *,
    user_name: str,
    text: str,
    num_hours: float,
) -> board_mirror.PostedComment:
    return board_mirror.PostedComment(
        user_name=user_name,
        text=text,
        created_at=hours_after_start(num_hours),
    )


class TestBoardMirror_Status(unittest.TestCase):

    def test_nobody_accepted_is_requested_reviewers(
        self,
    ):
        self.assertEqual(board_mirror.compute_status(make_activity()), board_mirror.Statuses.REQUESTED_REVIEWERS)

    def test_accepted_without_feedback_is_with_reviewer(
        self,
    ):
        activity = make_activity(assignee_user_names=("reviewer", ))
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.WITH_REVIEWER)

    def test_feedback_is_with_author(
        self,
    ):
        activity = make_activity(
            assignee_user_names=("reviewer", ),
            feedback_times=(hours_after_start(2), ),
        )
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.AUTHOR_ADDRESSING_FEEDBACK)

    def test_request_after_feedback_is_rerequest(
        self,
    ):
        activity = make_activity(
            feedback_times=(hours_after_start(2), ),
            review_request_times=(hours_after_start(0), hours_after_start(5)),
        )
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.WAITING_ON_REREVIEW)

    def test_new_feedback_after_rerequest_is_with_author(
        self,
    ):
        activity = make_activity(
            feedback_times=(hours_after_start(2), hours_after_start(7)),
            review_request_times=(hours_after_start(0), hours_after_start(5)),
        )
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.AUTHOR_ADDRESSING_FEEDBACK)

    def test_code_owner_request_on_ready_is_not_rerequest(
        self,
    ):
        ready_time = hours_after_start(5)
        activity = make_activity(
            feedback_times=(hours_after_start(2), ),
            ready_for_review_times=(ready_time, ),
            review_request_times=(hours_after_start(0), ready_time + datetime.timedelta(seconds=1)),
        )
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.AUTHOR_ADDRESSING_FEEDBACK)

    def test_request_well_after_ready_is_rerequest(
        self,
    ):
        activity = make_activity(
            feedback_times=(hours_after_start(2), ),
            ready_for_review_times=(hours_after_start(5), ),
            review_request_times=(hours_after_start(0), hours_after_start(6)),
        )
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.WAITING_ON_REREVIEW)

    def test_approved(
        self,
    ):
        activity = make_activity(
            is_approved=True,
            feedback_times=(hours_after_start(2), ),
        )
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.APPROVED)

    def test_draft_wins_over_everything(
        self,
    ):
        activity = make_activity(
            is_draft=True,
            is_approved=True,
            assignee_user_names=("reviewer", ),
        )
        self.assertEqual(board_mirror.compute_status(activity), board_mirror.Statuses.DRAFT)


class TestBoardMirror_ReviewByDate(unittest.TestCase):

    def test_newest_accept_review_by_the_reviewer(
        self,
    ):
        activity = make_activity(
            assignee_user_names=("reviewer", ),
            posted_comments=(
                make_comment(user_name="reviewer", text="/accept-review 2026-11-10", num_hours=1),
                make_comment(user_name="reviewer", text="/accept-review 2026-11-20", num_hours=3),
                make_comment(user_name="someone-else", text="/accept-review 2026-12-01", num_hours=4),
            ),
        )
        self.assertEqual(board_mirror.find_review_by_date(activity), datetime.date(2026, 11, 20))

    def test_invalid_date_falls_back_to_an_earlier_one(
        self,
    ):
        activity = make_activity(
            assignee_user_names=("reviewer", ),
            posted_comments=(
                make_comment(user_name="reviewer", text="/accept-review 2026-11-10", num_hours=1),
                make_comment(user_name="reviewer", text="/accept-review soon", num_hours=3),
            ),
        )
        self.assertEqual(board_mirror.find_review_by_date(activity), datetime.date(2026, 11, 10))

    def test_no_reviewer_gives_no_date(
        self,
    ):
        activity = make_activity(
            posted_comments=(make_comment(user_name="reviewer", text="/accept-review 2026-11-10", num_hours=1), ),
        )
        self.assertIsNone(board_mirror.find_review_by_date(activity))


class TestBoardMirror_Changes(unittest.TestCase):

    def test_matching_board_needs_no_change(
        self,
    ):
        self.assertIsNone(board_mirror.plan_board_change(make_activity()))

    def test_pr_not_on_board_is_skipped(
        self,
    ):
        self.assertIsNone(board_mirror.plan_board_change(make_activity(is_draft=True, board_item=None)))

    def test_status_change(
        self,
    ):
        board_change = board_mirror.plan_board_change(make_activity(is_draft=True))
        self.assertEqual(board_change.status_to_set, board_mirror.Statuses.DRAFT)
        self.assertFalse(board_change.review_by_date_changes)

    def test_stale_date_is_cleared(
        self,
    ):
        board_item = dataclasses.replace(BOARD_ITEM, review_by_date=datetime.date(2026, 11, 10))
        board_change = board_mirror.plan_board_change(make_activity(board_item=board_item))
        self.assertIsNone(board_change.status_to_set)
        self.assertTrue(board_change.review_by_date_changes)
        self.assertIsNone(board_change.review_by_date_to_set)


if __name__ == "__main__":
    unittest.main()
