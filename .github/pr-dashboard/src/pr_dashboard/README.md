# The `pr_dashboard` Package

What each module holds. For what the tools do and why, see the [main README](../../README.md).

## Modules

| Module | Holds |
|---|---|
| `pr_context.py` | The PR and what is attached to it: `PullRequest`, `User`, `RoleInRepo`, `Comment` and `Milestone`; and the `UpdatePlan` that describes changes to it |
| `pr_labels.py` | The label types and their categories, and `plan_update_from_pr`, which works out the labels of a PR |
| `pr_commands.py` | The `Commands`, how comments are parsed, the replies, and `plan_update_from_comment` |
| `load_labels.py` | Reads `labels.toml` into the label types |
| `board_mirror.py` | The board and its `Statuses`, the `PullRequestActivity` a status is worked out from, `compute_status`, `find_review_by_date`, and `plan_board_change` |
| `github_api.py` | Every call to GitHub: reads that return data, writes that return nothing, `apply_update_plan` and `apply_board_change` |
| `run_dashboard.py` | The script the workflows run: one action per workflow, and their command-line arguments |
