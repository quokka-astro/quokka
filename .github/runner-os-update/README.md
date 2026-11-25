# Runner OS Update

This tool keeps the operating system that CI runs on up to date. Every Quokka job asks for a fixed OS version in its `runs-on` label, such as `ubuntu-24.04`, so a new release from GitHub cannot break CI without warning. Once a week the tool checks which versions `ubuntu-latest` and `macos-latest` now run, and opens a PR that moves the fixed labels up to them. This README is for anyone reviewing or changing the code.

## Layout

| Path | Holds |
|---|---|
| `runner_os_update.py` | The code: raises the version in every `runs-on` label under `.github/workflows/` |
| `test_runner_os_update.py` | Offline tests; they never contact GitHub |
| `../workflows/runner-os-update*.yml` | The two GitHub Actions workflows that run the code and its tests |

## How a Run Works

1. Every Monday, or when someone starts it from the Actions tab, `runner-os-update.yml` starts a job on `ubuntu-latest` and one on `macos-latest`. Each reads its own OS version.
2. A third job runs `runner_os_update.py` with both versions. Every older `runs-on` label is raised; newer labels are never lowered.
3. If any label changed, the job opens a PR against `development`. CI on that PR is the first run on the new OS versions.
4. If the PR's CI fails, fixes are pushed to its branch, or it waits until the missing tools are published. Next week's run sees that the branch already exists and does nothing.

| Workflow | Runs when | Does |
|---|---|---|
| `runner-os-update.yml` | Every Monday, or started from the Actions tab | Opens the update PR |
| `runner-os-update-tests.yml` | A PR changes this folder or the tests workflow | Runs the offline tests |

## Running the Tests

From the repository root:

```bash
python3 -m unittest discover --start-directory .github/runner-os-update --pattern "test_*.py"
```

## Trying It Locally

From the repository root, the code rewrites the workflow files in place and prints each change, so `git diff` shows what the PR would contain:

```bash
python3 .github/runner-os-update/runner_os_update.py --ubuntu-version <version> --macos-version <version>
```

## Design Decisions

- **The PR is opened by the quokka-astro GitHub App,** rather than with the workflow's own token. GitHub does not run CI on PRs opened with that token, and that CI run is the whole point of the PR.
- **Labels are only ever raised.** A job that someone moved to a newer OS on purpose stays there.
- **`ubuntu-latest` and `macos-latest` labels are left alone.** Only the jobs that read the latest version use them.
- **The tool does not check whether the compilers and libraries CI installs exist for the new OS.** The install scripts check that themselves and say what is missing, so the failure shows up in the update PR's CI with a clear message.
