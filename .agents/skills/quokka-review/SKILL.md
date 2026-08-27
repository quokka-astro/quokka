---
name: quokka-review
description: "Review a Quokka pull request or GitHub issue — build and run the code to verify the claim, then post a short verdict comment. Analysis only; does not implement fixes."
---

# Quokka PR and Issue Review

Analyse and report. Never change source.

The argument is the target: a bare number is a pull request (`2158`), a number prefixed with `issue` is an issue (`issue 2110`). With no argument, ask which.

## Comments

Post every PR and issue comment with `--body-file`, never inline `--body "…"` — backticks and `$` in a review comment get command-substituted out of an inline body. End every comment with the fixed line `🤖 AI-assisted review`.

**Exception:** `/azp run …` CI triggers post raw with no footer; Azure Pipelines matches on exact command text.

## Setup (once per session)

```bash
# Must name a repo. If it errors, run `gh repo set-default quokka-astro/quokka` once.
gh repo set-default --view

# <review-dir> — record it; substitute literally, never as a shell variable
echo "$(git rev-parse --path-format=absolute --git-common-dir)/quokka-review"
```

## Isolation

Reviewing checks out other refs, so start from a clean tree:

```bash
git status --porcelain
```

Non-empty → stop and ask the user to commit, or to review in a separate worktree (`git worktree add ../review-<slug>`). Never stash their work.

Build both refs on throwaway branches keyed to this review's `<slug>`, not a commit hash, so concurrent reviews don't collide. Delete them when done: `git branch -D dev-<slug> pr-<slug>`.

## Reviewing a Pull Request

### Step 1 — Read and classify

```bash
gh pr view NNNN
gh pr diff NNNN
```

Derive a kebab-case `<slug>` from the title — `fix-amr-regrid` from "Fix AMR regrid bug".

| Type | Signal | Cycle |
|---|---|---|
| Bug fix | "fix", "closes issue #N", describes wrong existing behavior | RED → GREEN → EDGE |
| Feature | "add", "implement", "new", describes a new capability | SPEC → VERIFY → EDGE |

When in doubt use the **bug fix** cycle; it is stricter.

Scan the diff for `AMREX_GPU_DEVICE`, `amrex::ParallelFor`, or changes to `simulation.hpp` / `QuokkaSimulation.hpp` — those make the GPU build mandatory.

**Before the first `quokka config`, read `references/building.md`.** It sets up the environment, picks the preset, and holds the exact build commands.

---

## Bug Fix PRs: RED → GREEN → EDGE

```
NO CLAIM THAT A PR "FIXES" SOMETHING WITHOUT FIRST WATCHING IT FAIL ON THE BASE BRANCH
```

### Step 2 — RED

Build `origin/development` as `dev-<slug>`, then run with the parameters that expose the bug.

Document exactly what you observe: file count, error message, incorrect value. If the bug does not reproduce, stop and tell the user why — already fixed upstream, wrong parameters, platform difference.

### Step 3 — GREEN

Build the PR as `pr-<slug>` and run the identical scenario. Confirm the wrong behavior is gone, the run completes clean, and the output matches the PR's claim.

Add the GPU build if Step 1 flagged device code. A passing CPU build does not clear a PR that touches GPU kernels.

### Step 4 — EDGE

Run one scenario the PR should not affect — an explicit configuration, a different code path, an adjacent feature. Confirm it records the same value it records on the base branch.

Then go to **Reporting**.

---

## Feature PRs: SPEC → VERIFY → EDGE

### Step 2 — SPEC

Before building, write the contract from the PR description, linked issues, and diff:

- new inputs: parameters, TOML keys, function arguments
- outputs and effects
- stated boundary and error conditions

Infer from the diff where the description is vague, and note the gaps.

### Step 3 — VERIFY

Build the PR as `pr-<slug>`. Work the checklist — golden path first, then the stated boundaries. Mark every SPEC item verified, failed, or not-testable before writing the report.

Add the GPU build on the same rule as the bug-fix cycle.

### Step 4 — EDGE

Test one boundary the description does not cover, and run one existing adjacent test for regression.

Then go to **Reporting**.

---

## Reporting

Two files, two audiences. The **report** is the durable record, as detailed as the work warrants. The **comment** is read by a busy maintainer on a phone. Never post the report as the comment.

One rule governs the comment: **the author does not need convincing, they need locating.**

### 1. Report

```bash
ls <review-dir>/ 2>/dev/null | grep "^prNNNN-" || echo "none"
mkdir -p <review-dir>/prNNNN-<slug>
```

Write inside an existing `prNNNN-*` folder if there is one. Filename `<folder>-REVIEW.md`; follow-up rounds append `-v2`, `-v3`.

Contents: evidence table, one row per phase; analysis — root cause with `file:line` for bug fixes, design match for features; PR-introduced concerns; pre-existing issues. Derivations, parameter sweeps and raw numbers live here.

### 2. Comment

A separate file, `<folder>-COMMENT.md`. Fill this and delete every empty section:

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

🤖 AI-assisted review
```

1. **Verdict first.** Line 1 is the conclusion a maintainer acts on.
2. **≤ 300 words outside `<details>`.** Count before posting.
3. **One finding = one line.** Bold claim, `file:line`, then at most one sentence of trigger-and-result.
4. **Cap at 5**, worst first, split Blocking / Non-blocking. Drop any finding whose fix would not change what the author does. Pre-existing issues get one Non-blocking line, or none.
5. **Numbers, not adjectives.** `error norm 1.33e-09 vs tol 3e-03`, not "well within tolerance". Paste the real abort message.
6. **No preamble, no recap, no closer.** Not "I reviewed…", "Overall this is a nice change", "Let me know if…". Start with the verdict, end with the last finding.
7. **"None" is a complete section.** Write `**Blocking:** none.` and stop.

**One finding, before and after** — same concern, same evidence, from a real review.

Bad, 118 words:

> **1. `num_periods` has no input validation; non-positive values make the test pass vacuously.** `setup.num_periods=0` and `setup.num_periods=-1` both produce `stopTime_ <= 0`, run zero timesteps, and exit 0 with "error norm 6.368370e-18 (tol = 2.000000e-03)" — a green PASS for a simulation that never advanced. A negative value is worse than a no-op: it turns a correctness test into an unconditional success. These same files already validate their other setup parameters (`testFieldLoop.cpp:234` aborts on `loop_radius <= 0`), so guarding is the established convention here and the new parameter is the odd one out. Suggest an `amrex::Abort` in each of the seven sites.

Good, 34 words:

> 1. **`num_periods` unvalidated** — all 7 sites. `setup.num_periods=0` or `=-1` runs zero timesteps and reports PASS at error norm 6.37e-18; these files already abort on bad `loop_radius` (`testFieldLoop.cpp:234`).

Same claim, same `file:line`, same number, same fix — implied by "unvalidated". A five-finding review written this way lands near **165 words**, so the 300 cap binds only on genuinely large reviews.

Before posting, check: reading **only** the Verdict line, does a maintainer know whether to merge? If not, rewrite that line.

**Follow-up rounds (v2, v3, …):** re-read the code at every carried-over `file:line` and drop findings whose code is gone. A round where everything was fixed is three lines — verdict, what you re-ran, what still stands.

### 3. Post

```bash
gh pr comment NNNN --body-file <review-dir>/<folder>/<folder>-COMMENT.md
```

Then a 1–2 sentence summary in chat.

---

## Reviewing a GitHub Issue

```bash
gh issue view NNNN
```

Derive a `<slug>` from the title. Inspect the affected source files and check git log for related commits and PRs. If the root cause is clear from code and history, that is enough; otherwise build `origin/development` as `dev-<slug>` per `references/building.md` and reproduce.

For audit issues — labels `code-audit` / `codex` — verify each claimed finding against the **current** code, and state per finding: still valid, or invalid and why.

Same two files. Report to `<review-dir>/issueNNNN-<slug>/issueNNNN-<slug>.md`: root cause, fix location(s), latent concerns, and whether the issue is resolved or still open. Then `issueNNNN-<slug>-COMMENT.md`, under all the comment rules above:

```markdown
**Verdict:** <root cause in one line | still open | already fixed by #NNNN>
**Fix location:** `file.cpp:NN` <what changes>

**Also worth knowing**
- <one line each, at most 3>

🤖 AI-assisted review
```

For audit issues, one line per claimed finding: `**Finding N** — valid / invalid because X`.

```bash
gh issue comment NNNN --body-file <review-dir>/issueNNNN-<slug>/issueNNNN-<slug>-COMMENT.md
```

Then a 1–2 sentence summary in chat.
