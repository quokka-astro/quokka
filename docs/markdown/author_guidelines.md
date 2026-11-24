# Author Guidelines

These guidelines are for you as the author of a pull request (PR), and set out what to do from your first draft to the merge. For how to set up a branch and build the code, see the [Contributing Guide](contributing.md). For the policy that PRs are held to, see the [Maintainers Guide](maintainer_guidelines.md).

---

## From Draft to Merge

1. **Open your PR as a draft** while you work on it.
2. **Get your PR ready for a reviewer:**
    - Complete the PR description, and make sure the checks pass.
    - If your change affects code that runs on GPUs, comment `/azp run quick` and `/azp run rocm-quick` for changes to test problems, or `/azp run` for changes to shared code. If you do not have write access, ask a maintainer to run them.
    - Add a `change/<value>` label to say why you made this change, and any other labels that apply.
    - If your PR is more or less urgent than usual, say so with `/set-priority <low|high>`.
3. **Mark your PR ready for review,** and comment `/request-review`.
4. **Respond to the review.** Answer each thread, make any changes, and resolve the thread once you have addressed it. If you decide against a suggestion, say why before you resolve it.
5. **Ask for another look** once every thread is resolved, by commenting `/request-review` again. Repeat steps 4 and 5 until your PR is approved.
6. **Merge your approved PR** by clicking **Merge when ready**. This is your responsibility, since you know when your PR is finished. If you do not have write access, your reviewer does this for you.

> **Note: A push removes an approval**
>
> Any push to an approved PR dismisses the approval, including a rebase, so your PR needs approving again. Finish your changes before asking for the final approval.
>

---

## Labels

Each label is written as `<category>/<value>`, and hovering over a label on GitHub shows its description.

### Labels Set for You

| Category | What it says | How it is set |
|---|---|---|
| `scope/<value>` | Which parts of the code your PR changes | From the files you changed |
| `size/<value>` | How many lines your PR changes: `XS` (under 10), `S` (10 to 29), `M` (30 to 99), `L` (100 to 499), `XL` (500 or more) | From the number of lines you changed |

These labels are set when your PR is opened, and updated on every push once it is out of draft.

### Labels for You to Add

These labels tell your reviewer and the maintainers what cannot be worked out automatically. Add every one that applies to your PR, starting with `change/`. If you do not have permission to add labels, ask your reviewer.

| Category | What it tells them |
|---|---|
| `change/<value>` | Why you made this change: `feature`, `fix`, `performance`, `refactor` or `maintenance` |
| `priority/<value>` | How soon your PR should be reviewed, if not at the normal pace: `low` or `high`. Set it with `/set-priority` |
| `problem/<value>` | What kind of problem your PR addresses |
| `status/<value>` | Something that affects whether your PR can merge |
| `platform/<value>` | A platform your PR affects |
| `agent/<value>` | An AI agent that found or wrote the work |

---

## Commands

Comment `/help` on any PR to list the commands. A command only runs at the start of a comment. A thumbs-up on your comment means it ran; if it did not run, a reply says why.

---

## When Your PR Is Labelled `scope/undefined`

This label means a file in your PR does not belong to a scope yet. To resolve it:

1. **Open a separate PR** that adds the path to its scope in `.github/pr-dashboard/labels.toml`:

    ```diff
     value = "<scope>"
     ...
         "src/<existing-folder>/*",
    +    "src/<new-folder>/*",
    ```

2. **Once that PR has merged,** update your branch from `development`.
3. **If the file was committed by accident,** remove it.
4. **If no scope fits, or you are unsure,** reach out to the team.
