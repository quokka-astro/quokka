---
name: quokka-review
description: "Use when reviewing a Quokka pull request or GitHub issue — classify as bug fix (red-green cycle) or feature (spec-verify cycle), or analyse a GitHub issue root cause. Not for implementing fixes."
---

# Quokka PR and Issue Review

## When NOT to Use

- Implementing a fix for a GitHub issue — this skill analyses and reports only
- Making any source code change

## PR Comment Attribution

**REQUIRED:** Post every PR and issue comment with `--body-file`, **never** inline `--body "…"`. Review comments are dense with backticked `file.cpp:NN` citations and `$` in error norms, and the shell command-substitutes both out of an inline body. If you build the file with a heredoc rather than a file-write tool, quote the delimiter (`<< 'EOF'`) to stop the same expansion inside it.

End every comment with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. The file you post is the short **comment** file, never the full report — see **Reporting** below.

**Exception:** CI trigger comments (`/azp run rocm-quick`) must be posted raw with no attribution footer — Azure Pipelines matches on exact command text and will fail if the body contains anything else.

## Environment Setup (once per session)

Run all three from the repo root:

```bash
# 1. Tooling — `quokka` must match the repo copy. bootstrap.sh installs it into
#    ~/.local/bin, but SKIPS an install that is already on PATH, so check for staleness.
command -v quokka >/dev/null || ./scripts/bash/bootstrap.sh
cmp -s scripts/bash/quokka "$(command -v quokka)" \
    && echo 'quokka up to date' \
    || echo 'STALE: run  install -m755 scripts/bash/quokka ~/.local/bin/quokka'

# 2. GitHub — which repo does `gh` resolve to?
gh repo set-default --view

# 3. Paths — record both
echo "repo-root:  $(git rev-parse --show-toplevel)"
echo "review-dir: $(git rev-parse --path-format=absolute --git-common-dir)/quokka-review"
```

**On step 1:** a stale `quokka` mis-handles `--source default` — either rejecting the word outright or aborting when the rc is absent instead of warning. Refresh it before going on.

**On step 2:** no `gh` command in this skill passes `--repo`, so they all work in a fork. A clone with a single `origin` needs nothing here. A clone with **several** remotes and no default set makes `gh` prompt for the base repo, which hangs a non-interactive session — if step 2 errors instead of naming a repo, run `gh repo set-default quokka-astro/quokka` once and move on.

**On step 3:** substitute both literally wherever `<repo-root>` and `<review-dir>` appear below — shell state does not survive between commands, so neither can be carried in a variable.

- **`<repo-root>`** goes on every `quokka` command as `--root <repo-root>`. **Keep it there.** It is what makes each command independent of the current directory, so nothing breaks when you `cd` into `tests/` to run a binary or into a build directory to inspect output. Drop it and `quokka` falls back to the cwd, then aborts with `tests directory not found: … (use --root to specify the quokka root)`.
- **`<review-dir>`** is where report files go. It sits inside the clone's shared `.git`, so reports are never committed, never appear in `git status`, need no `.gitignore` entry, and stay reachable from every worktree of the clone.

**Running a binary directly.** `quokka run` already executes from `<repo-root>/tests`. When you bypass it to pass bug-exposing parameters by hand, `cd <repo-root>/tests` first. That whole directory is gitignored, so plotfiles, checkpoints, slices and CSVs land somewhere harmless. Run from the repo root instead and `slice*` and `*.csv` are **not** covered by the root `.gitignore` — they show up as untracked files and trip the clean-tree check above on the next review.

**Build environment.** Every `quokka` command below passes `--source default`, which sources `~/.config/quokka/quokka.rc` — the per-machine place for `module load`, or a CUDA `bin` prepended to `PATH` so cmake finds `nvcc`. Machines needing none of that simply leave the file absent; `--source default` then prints `Warning: default environment file ... not found` and carries on. **That warning is expected, not a failure** — pass the flag unconditionally.

**Isolation.** Reviewing checks out other refs, so it must not disturb the developer's working tree. Confirm the tree is clean before starting:

```bash
git status --porcelain
```

Non-empty output → stop and ask the user to commit, or to run the review in a separate worktree (`git worktree add ../review-<slug>`). Never stash their work.

Then build both the PR and its base from **uniquely-named throwaway branches**, keyed to this review's `<slug>` — **not** a commit hash — so concurrent reviews of the same base don't collide. Run `git submodule update --init --recursive` after each checkout so submodule pins match, and delete the branches when done (`git branch -D dev-<slug> pr-<slug>`).

## Reviewing a Pull Request

### Step 1 — Read the PR and classify

```bash
gh pr view NNNN
gh pr diff NNNN
```

Derive a short kebab-case slug from the PR title (e.g., `fix-amr-regrid` from "Fix AMR regrid bug") — used for the artifact folder name when no prior folder exists.

**Classify the PR type** — this determines which review cycle to follow:

| Type | Signal | Cycle |
|---|---|---|
| Bug fix | "fix", "closes issue #N", describes wrong existing behavior | RED → GREEN → EDGE |
| Feature / enhancement | "add", "implement", "new", describes a new capability | SPEC → VERIFY → EDGE |

When in doubt, use the **bug fix** cycle (stricter).

**Check for GPU device code** — scan the diff for `AMREX_GPU_DEVICE`, `amrex::ParallelFor`, or changes to core templates (`simulation.hpp`, `QuokkaSimulation.hpp`).

**Choose the preset yourself — never assume a fixed machine.** This skill runs in several environments, and they differ in both toolchain and device:

| Environment | Toolchain | GPU device |
|---|---|---|
| macOS | Apple clang | none |
| Ubuntu container on macOS | `nvcc` | none |
| Ubuntu container on Linux | `nvcc` | yes |
| Docker container on Linux | ROCm / `hipcc` | yes |

Detect what is actually present, then decide using these rules:

- **Dimensionality** — run the test at the **lowest** dimensionality the problem allows. Only move up to `2d` / `3d` when the problem genuinely requires it.
- **On macOS, use the `-apple` suffix** (`1d-apple`, `2d-apple`, `3d-apple`). Build settings are identical to the bare preset; the only difference is a separate `build/<Nd>-apple` directory, which keeps an Apple-clang build from clobbering the `nvcc` build tree when a container shares the same repo.
- **Use the GPU when it pays** — if a GPU **device** is available and the test is slow on CPU, run it with the GPU preset.
- **GPU correctness is mandatory for GPU-touching changes** — if the change touches GPU functions or variables, **always** build the problem with the GPU preset when a toolchain is available. Where there is a toolchain but no device, the compile alone is the check (it is exactly what catches device-lambda capture errors) and runtime behavior is CI's job.

Detect toolchain and device **separately** — the toolchain decides whether you can compile, the device decides whether you can run. Note `nvcc` may not be on the interactive PATH, so check install locations too:

```bash
# toolchain → GPU preset suffix
if [ -x /usr/local/cuda/bin/nvcc ] || command -v nvcc &>/dev/null; then
    echo "CUDA toolkit present → GPU preset suffix: -cuda"
elif command -v hipcc &>/dev/null || [ -d /opt/rocm ]; then
    echo "ROCm present → GPU preset suffix: -hip"
else
    echo "No GPU toolchain → skip GPU build steps"
fi

# device → can you actually RUN on the GPU, or only compile?
nvidia-smi -L 2>/dev/null || rocm-smi --showid 2>/dev/null || ls /dev/kfd 2>/dev/null \
    || echo "No GPU device → compile-only check"
```

Record both results and use the suffix in the GPU build steps below.

---

## Bug Fix PRs: RED → GREEN → EDGE

**Core principle:** If you didn't watch the test fail on the base branch, you don't know whether you're testing the right thing.

### The Iron Law

```
NO CLAIM THAT A PR "FIXES" SOMETHING WITHOUT FIRST WATCHING IT FAIL ON THE BASE BRANCH
```

### Step 2 — RED: Reproduce the bug on the base branch

Always explicitly check out the base before building — do not assume the working tree is already on the right ref. Build it from a uniquely-named throwaway branch off `origin/development` (a throwaway branch keeps the checked-out ref undisturbed and avoids collisions with concurrent reviews):

```bash
git fetch origin development
git checkout -B dev-<slug> origin/development
git submodule update --init --recursive
```

`quokka config` must be run after every branch checkout or submodule update.

```bash
quokka config -d <preset> --delete --source default --root <repo-root> -DQUOKKA_PYTHON=OFF
quokka build -d <preset> <Target> --source default --root <repo-root>
# then run with the parameters that expose the bug, from the gitignored tests dir.
# Both paths must be absolute — the binary resolves nothing relative to tests/:
#   cd <repo-root>/tests
#   <repo-root>/build/<preset>/src/problems/<Target>/<Target> \
#       <repo-root>/inputs/<Target>.toml <param>=<value>
```

**Confirm the wrong behavior manifests.** Document exactly what you observe (file count, error message, incorrect value, etc.).

If the bug does **not** reproduce, stop — note this to the user and explain why (already fixed upstream, wrong parameters, platform difference).

**Never skip this step.** A test that passes immediately proves nothing.

### Step 3 — GREEN: Check out the PR branch and validate the fix

Use `gh pr checkout`, **not** `git fetch origin <pr-branch>` — most community PRs come from forks, whose head branch does not exist on `origin` (`fatal: couldn't find remote ref …`). `gh` resolves the upstream repo itself, so this also works when the developer's `origin` is their own fork. `--force` resets the branch a previous review round left behind.

```bash
gh pr checkout NNNN --branch pr-<slug> --force
git submodule update --init --recursive
quokka config -d <preset> --delete --source default --root <repo-root> -DQUOKKA_PYTHON=OFF
quokka build -d <preset> <Target> --source default --root <repo-root>
# run the identical scenario from Step 2
```

Confirm:
- The previously observed wrong behavior is gone
- The run completes without errors
- Output matches what the PR description claims

**GPU build (required if PR touches GPU device code — see Step 1)**

Take the dimensionality preset (e.g. `3d`) and append the detected suffix:

```bash
quokka config -d <Nd>-<suffix> --delete --source default --root <repo-root> -DQUOKKA_PYTHON=OFF
quokka build -d <Nd>-<suffix> <Target> --source default --root <repo-root>
```

This catches device-code restrictions and GPU lambda capture errors that only surface during GPU compilation. A CPU build that passes does not clear the PR if the PR touches GPU kernels.

### Step 4 — EDGE: Verify unchanged behavior

Test at least one scenario the PR should **not** affect — an explicit configuration, a different code path, or an adjacent feature. Confirm it still produces correct results.

Then go to **Reporting** below.

### Red flags — restart from Step 2

- Went straight to the PR branch without reproducing on base
- "The PR looks correct so I'll skip running it"
- Test passed immediately on the base branch with no investigation
- Only checked one scenario (no edge-case step)

---

## Feature PRs: SPEC → VERIFY → EDGE

### Step 2 — SPEC: Extract the contract

Do **not** build yet. From the PR description, linked issues, and diff, write down:
- What new inputs does the feature accept (parameters, TOML keys, function arguments)?
- What outputs or effects does it produce?
- What are the stated boundary or error conditions?

This becomes your explicit test checklist. If the PR description is vague, infer from the diff and note any gaps.

### Step 3 — VERIFY: Check out the PR branch and exercise the feature

```bash
gh pr checkout NNNN --branch pr-<slug> --force
git submodule update --init --recursive
quokka config -d <preset> --delete --source default --root <repo-root> -DQUOKKA_PYTHON=OFF
quokka build -d <preset> <Target> --source default --root <repo-root>
```

Work through each item in your SPEC checklist:
- Run the scenario that exercises the claim
- Confirm the output matches what the PR describes

Cover the golden path first, then stated boundary conditions.

**GPU build** — same rule and commands as in the bug-fix cycle: required if the PR touches GPU device code and a toolchain was detected in Step 1.

### Step 4 — EDGE: Stress test and regression check

- Test at least one boundary the PR description does not explicitly cover
- Run at least one existing adjacent test to confirm no regression

Then go to **Reporting** below.

---

## Reporting (both cycles)

**Two artifacts, two audiences.** The report file is the durable record — keep it as detailed as the work warrants. The PR comment is read by a busy maintainer on a phone — it is short by construction. **Never post the report file as the comment.** That is what makes review comments unreadable.

### 1. Write the report file (full detail)

Substitute `<review-dir>` (Environment Setup step 3) and the real PR number for `NNNN` before running:

```bash
# Reuse the folder a prior review round created, if there is one
ls <review-dir>/ 2>/dev/null | grep "^prNNNN-" || echo "none"
mkdir -p <review-dir>/prNNNN-<slug>
```

If a `prNNNN-*` folder already exists, write the report inside it rather than creating a second one. The report filename is `<folder>-REVIEW.md` (e.g. `pr2020-migrate-datatable-hdf5-REVIEW.md`). Follow-up reviews append `-v2`, `-v3`, etc.

Contents — evidence table (one row per phase), analysis (root cause + `file:line` for bug fixes; design match for features), PR-introduced concerns, pre-existing issues. Long derivations, parameter sweeps, and raw numbers belong **here**, not in the comment.

### 2. Write the comment file (short by construction)

Write a **separate** file `<folder>-COMMENT.md`. Fill this template and delete every empty section:

```markdown
**Verdict:** <ship it | N blocking | cannot verify X — one line, no hedging>
**Checked:** <preset + targets built, tests run, the one or two headline numbers>

**Blocking**
1. **<what breaks>** — `file.cpp:NN`. <Trigger → result. One sentence.>

**Non-blocking**
- **<claim>** — `file.cpp:NN`. <One clause.>

<details><summary>Evidence</summary>

| Phase | Ran | Observed |
|---|---|---|
| … | … | … |

</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

**Hard rules:**

1. **Verdict first.** Line 1 is the conclusion a maintainer acts on. Not context, not what you did, not "I reviewed this PR."
2. **≤ 300 words outside `<details>`.** Count before posting. Over budget means cut findings or move detail into `<details>`, never shrink the font of the argument.
3. **One finding = one line.** Bold claim, `file:line`, then at most one sentence of trigger-and-result. Supporting tables, derivations, and alternative fixes go in the report file.
4. **Cap at 5 findings**, ranked worst first, split Blocking / Non-blocking. More than five means you are listing, not reviewing — keep the five that change what the author does.
5. **Numbers, not adjectives.** `error norm 1.33e-09 vs tol 3e-03` beats "well within tolerance". Paste the actual abort message, not a paraphrase.
6. **No preamble, no recap, no closer.** Forbidden: "I reviewed…", "Overall this is a nice change", "Let me know if…", "Happy to re-review". Start with the verdict, end with the last finding.
7. **Evidence collapses.** The phase table goes inside `<details>`; only the headline numbers appear in `Checked:`.
8. **Suppress tangents.** Pre-existing issues get at most one line under Non-blocking, or are dropped. Never a section of their own in the comment.
9. **"None" is a complete section.** No concerns means write `**Blocking:** none.` and stop — do not pad with praise.

**One finding, before and after** — same concern, same evidence, from a real review:

Bad (118 words, a paragraph of reasoning):

> **1. `num_periods` has no input validation; non-positive values make the test pass vacuously.** `setup.num_periods=0` and `setup.num_periods=-1` both produce `stopTime_ <= 0`, run zero timesteps, and exit 0 with "error norm 6.368370e-18 (tol = 2.000000e-03)" — a green PASS for a simulation that never advanced. A negative value is worse than a no-op: it turns a correctness test into an unconditional success. These same files already validate their other setup parameters (`testFieldLoop.cpp:234` aborts on `loop_radius <= 0`), so guarding is the established convention here and the new parameter is the odd one out. Suggest an `amrex::Abort` in each of the seven sites.

Good (34 words):

> 1. **`num_periods` unvalidated** — all 7 sites. `setup.num_periods=0` or `=-1` runs zero timesteps and reports PASS at error norm 6.37e-18; these files already abort on bad `loop_radius` (`testFieldLoop.cpp:234`).

Same claim, same `file:line`, same number, same suggested fix — implied by "unvalidated". The argument for *why it matters* belongs in the report file; the author does not need convincing, they need locating. A full five-finding review written this way lands around **165 words**, so the 300 cap binds only on genuinely large reviews.

**Pre-send check** — delete from the comment:
- Any sentence describing what you were about to do or just did.
- Any finding whose fix the author would not change behaviour over.
- Any restatement of the PR description back at its own author.
- Any hedging adverb carrying no information ("somewhat", "arguably", "it seems").

Then verify: reading **only** the Verdict line, does a maintainer know whether to merge? If not, rewrite that line.

**Follow-up reviews (v2, v3, …):** Before listing findings carried over from a prior round, re-read the code at each cited `file:line` to confirm the issue still exists — commits between rounds may have already fixed it. Drop any finding whose code is gone. A follow-up where everything was fixed is *three lines*: verdict, what you re-ran, what still stands.

### 3. Post

Post the **comment** file. The report file stays local — it is the working record for follow-up rounds, not a publication.

```bash
gh pr comment NNNN --body-file <review-dir>/<folder>/<folder>-COMMENT.md
```

Give a **1-2 sentence summary in chat** — full detail is in the report file.

---

## Output Analysis: What to Ignore, What to Weigh

When reading test output, **ignore** AMReX performance advisories such as:

```
[Warning] [Performance] The grid blocking factor (N) is too small for reasonable performance.
It should be 32 (or greater) when running on GPUs, and 16 (or greater) when running on CPUs.
```

These reflect grid-layout preferences, not test correctness. Do not mention them in the report unless the PR is specifically about grid decomposition performance.

**Focus on:** convergence rates, error norms, SUCCESS/FAIL lines, and actual error/abort messages.

**Judging a marginal failure:** when a test fails with an error norm *close* to its tolerance (say within a factor of ~2), do not immediately blame the PR. Check whether the norm also sits near the tolerance on the base branch — if base and PR give nearly identical norms and both hover at the threshold, you are looking at tolerance noise (platform/compiler sensitivity), which is a pre-existing-issue note, not a PR-introduced concern. A norm that jumps by an order of magnitude between base and PR branch is a real regression regardless of whether it technically still passes.

---

## Reviewing a GitHub Issue

Use when the user says "review issue NNNN" (analysis only, no fix).

### 1. Read the issue

```bash
gh issue view NNNN
```

Derive a short kebab-case slug from the issue title (e.g., `amr-level-drop` from "AMR level drops to 0") — use it in all artifact filenames for this review.

### 2. Analyse

Read the issue, inspect the affected source files, and check git log for related commits and PRs.

For AI-generated audit issues (labels `code-audit` / `codex`), verify each claimed finding against the **current** code before accepting it — audit findings are written against a snapshot and may already be fixed or partially wrong. State per finding: still valid, or invalid and why.

- If the root cause is clear from code/history alone, that is sufficient.
- If not obvious, reproduce on a throwaway base branch off `origin/development`, then build:

```bash
git fetch origin development
git checkout -B dev-<slug> origin/development
git submodule update --init --recursive
```

### 3. Write report and post comment

Same two-file split as a PR review — the report holds the detail, the comment is short.

Write findings to `<review-dir>/issueNNNN-<slug>/issueNNNN-<slug>.md`: root cause, fix location(s), latent concerns, and whether the issue is resolved or still open.

Then write `issueNNNN-<slug>-COMMENT.md` and post **that**. All the comment rules from **Reporting** apply — verdict first, ≤ 300 words, one line per finding, evidence in `<details>`:

```markdown
**Verdict:** <root cause in one line | still open | already fixed by #NNNN>
**Fix location:** `file.cpp:NN` <what changes>

**Also worth knowing**
- <one line each, at most 3>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

For AI-generated audit issues, one line per claimed finding: `**Finding N** — valid / invalid because X`. Do not restate the finding text back at the issue.

```bash
gh issue comment NNNN --body-file <review-dir>/issueNNNN-<slug>/issueNNNN-<slug>-COMMENT.md
```

Give a **1-2 sentence summary in chat** — the full detail is in the report file.
