import dataclasses
import enum


class RoleInRepo(enum.StrEnum):
    READ = "read"
    TRIAGE = "triage"
    WRITE = "write"
    MAINTAIN = "maintain"
    ADMIN = "admin"


@dataclasses.dataclass(frozen=True)
class Milestone:
    milestone_number: int
    title: str


@dataclasses.dataclass(frozen=True)
class PullRequest:
    pr_number: int
    author_user_name: str
    assignee_user_names: tuple[str, ...]
    label_names: tuple[str, ...]
    num_lines_changed: int
    edited_file_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class User:
    user_name: str
    role_in_repo: RoleInRepo


@dataclasses.dataclass(frozen=True)
class Comment:
    comment_id: int
    user: User
    text: str


@dataclasses.dataclass(frozen=True)
class MilestoneChange:
    milestone_number: int | None


@dataclasses.dataclass(frozen=True)
class UpdatePlan:
    label_names_to_add: tuple[str, ...] = ()
    label_names_to_remove: tuple[str, ...] = ()
    assignee_user_names_to_add: tuple[str, ...] = ()
    assignee_user_names_to_remove: tuple[str, ...] = ()
    milestone_change: MilestoneChange | None = None
    milestone_title_to_create: str | None = None
    reply: str | None = None
    react: bool = False
