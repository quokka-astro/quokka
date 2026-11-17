# The `pr_dashboard` Package

What each module holds. For what the tools do and why, see the [main README](../../README.md).

## Modules

| Module | Holds |
|---|---|
| `pr_context.py` | The PR and what is attached to it: `PullRequest`, `User`, `RoleInRepo`, `Comment` and `Milestone`; and the `UpdatePlan` that describes changes to it |
| `pr_labels.py` | The label types and their categories, and `plan_update_from_pr`, which works out the labels of a PR |
| `pr_commands.py` | The `Commands`, how comments are parsed, the replies, and `plan_update_from_comment` |
| `load_labels.py` | Reads `labels.toml` into the label types |
| `github_api.py` | Every call to GitHub: reads that return data, writes that return nothing, and `apply_update_plan` |
| `run_dashboard.py` | The script the workflows run: one action per workflow, and their command-line arguments |
