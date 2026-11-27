# PR Dashboard Tools

These tools keep the labels on Quokka PRs in step with what each PR changes, and let people act on a PR by writing commands in its comments. `/help` on any PR lists the commands, and each label has a description on GitHub. This README is for anyone reviewing or changing the code.

## Layout

| Path | Holds |
|---|---|
| `labels.toml` | The labels the tools manage, and the rules for giving them to PRs |
| `src/pr_dashboard/` | The code; see [its README](src/pr_dashboard/README.md) for what each module holds |
| `tests/` | Offline tests; they never contact GitHub |
| `../workflows/pr-dashboard-*.yml` | The four GitHub Actions workflows that run the code |

## How a Run Works

Every run follows the same steps:

1. A GitHub event starts one of the workflows.
2. The workflow runs one action of `run_dashboard.py`, with every input given explicitly on the command line.
3. The code reads what it needs from GitHub and decides what should change, as an `UpdatePlan`: labels to add or remove, assignees, a milestone, a reply, a reaction.
4. `apply_update_plan` writes that plan to GitHub.

Deciding and writing are kept apart, so every decision can be tested offline and inspected with `--dry-run`, and every run prints its plan to the Actions log.

| Workflow | Runs when | Action |
|---|---|---|
| `pr-dashboard-label-pr.yml` | A PR is opened, reopened or marked ready for review, or pushed to while out of draft | `label-pr`: sets the scope and size labels |
| `pr-dashboard-run-command.yml` | A comment containing `/` is posted on a PR | `run-command`: runs a command, or explains why it was not run |
| `pr-dashboard-update-label-definitions.yml` | `labels.toml` changes on `development`, or someone starts it from the Actions tab | `update-label-definitions`: creates and updates the label names, colours and descriptions on GitHub |
| `pr-dashboard-tests.yml` | Any PR is opened or updated | Runs the offline tests |

## Changing the Labels

Edit `labels.toml`. Each category sets a prefix, and each label holds only its value; the label name on GitHub is the prefix plus the value, for example `scope/` and `hydro`.

- Scope labels list `path_patterns`, and `*` also matches `/`. A PR matches every scope with a pattern matching one of its changed files.
- Scope labels also set a `relevance_level`, where 1 is the most relevant. Of the scopes a PR matches, it only gets those on the most relevant level, so less relevant changes, such as tests that come with a physics change, do not add a label. Scopes on the same level all show, and with every scope on level 1 a PR gets every scope it matches.
- Size labels list `min_num_lines`. A PR gets the largest one its line count reaches.
- The priority `default_value` has no label: a PR with no priority label has that priority, and `/set-priority <default_value>` removes the priority label.
- A changed file that matches no scope gives the PR the `scope/undefined` label, so a missing pattern shows up on the PR.
- A file that should have no scope goes in `unlabelled_path_patterns`. It gives the PR no scope label, and does not count as unmapped.

After the change merges into `development`, the label workflow creates any new labels and updates colours and descriptions. A PR that is already open keeps its old labels until it is labelled again, on its next push or with `/recompute-labels`.

## Running the Tests

The tools need Python 3.11 or newer and `gh` 2.48 or newer. From the repository root:

```bash
PYTHONPATH=.github/pr-dashboard/src python3 -m unittest discover --start-directory .github/pr-dashboard/tests --pattern "test_*.py"
```

One test checks that every file in the repository has a scope or is listed in `unlabelled_path_patterns`. This keeps the whole codebase accounted for, so the scope labels always show what a PR changes. A PR that adds a folder without a pattern for it fails this test until the folder is given a scope or left out on purpose.

## Debugging a PR

The actions can be run from a local checkout with a `gh` login. `--dry-run` prints the plan without writing anything:

```bash
PYTHONPATH=.github/pr-dashboard/src python3 -m pr_dashboard.run_dashboard label-pr --repo quokka-astro/quokka --pr <number> --dry-run
PYTHONPATH=.github/pr-dashboard/src python3 -m pr_dashboard.run_dashboard update-label-definitions --repo quokka-astro/quokka --dry-run
```

## Design Decisions

Decisions I made along the way, and why.

### Labels

- **The tools only manage the `scope/`, `size/` and `priority/` labels,** rather than every label on a PR. Labels without these prefixes are left alone. `scope/` and `size/` labels edited by hand are replaced when the labels are recalculated, and `priority/` labels are only changed with `/set-priority`.
- **Normal priority has no label,** rather than a `priority/normal` label on almost every PR. A priority label only appears when a PR is more or less urgent than usual.
- **A file with no matching scope gives the PR the `scope/undefined` label,** rather than a fallback scope. The label shows where `labels.toml` is missing a pattern. It shows whatever the relevance levels.
- **Relevance levels decide which matched scopes show,** rather than which files are mapped. Every file still needs a scope, so the test that the whole codebase is mapped still holds, and the levels only reduce how many labels a PR shows. Each scope has a fixed level, so the labels of a PR stay predictable from its changed files.

### Commands

- **Commands only run at the start of a comment,** rather than anywhere in it, so a command quoted or mentioned in passing is not run. A command found elsewhere gets a reply saying how to run it.
- **Every command gets visible feedback,** rather than failing silently. A thumbs-up means it ran; a reply says why it did not. Misspelt commands are ignored, since other tools on Quokka also use `/` commands.
- **One reviewer holds a review at a time,** rather than everyone asked to review, so it is clear who is responsible. `/accept-review` assigns the commenter, and refuses while someone else holds the review.
- **The date given to `/accept-review` is read as a local date,** rather than a UTC date, since Quokka is maintained from around the world. A date up to one day in the past is accepted.
- **`/request-review` only goes through once the PR is ready,** rather than whenever it is asked: not a draft, every check passed or skipped, and every review thread resolved. The author resolves a thread once they have addressed it, so the reviewer gets back a clean PR. The request goes to the reviewer who accepted the review, or to the team if nobody has. It is the same command the first time and every time after.
- **`/request-review` trusts every check and every thread,** rather than a chosen list. Anything allowed to act on a PR is considered by its author; a flaky or unhelpful check is fixed or removed where it runs, not worked around here.

### Running the Code

- **The tools only use what GitHub already tracks,** rather than keeping their own records. Each run reads the PR from GitHub and writes its changes back as labels, assignees, milestones, reactions and replies.
- **The labels workflow uses `pull_request_target`,** rather than `pull_request`, so it can label PRs from forks. It is safe because it checks out only these tools from the base branch and never runs code from the PR.
