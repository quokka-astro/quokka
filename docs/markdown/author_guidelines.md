# Author Guidelines

These guidelines are for you as the author of a pull request (PR). They walk through what happens from your first draft to the merge: which steps are yours, which are done for you, and what you get back at each stage. Following them means your PR reaches a reviewer ready to be reviewed, you can always see who has it and by when, and you know what is needed for it to merge. For how to set up a branch and build the code, see the [Contributing Guide](contributing.md). For the policy that PRs are held to, see the [Maintainers Guide](maintainer_guidelines.md).

> **Tip: The comment section is the command line of your PR**
>
> Writing `/help` as a comment on any PR lists every command. A change made with a command shows in the conversation who made it and when, so anyone reading your PR can follow how it got to where it is.
>

---

## From Draft to Merge

1. **Open your PR as a draft** while you work on it.
    - **Done for you:** your PR is given scope, size and priority labels, and the automated checks run on every push.
2. **Get your PR ready for a reviewer.** Complete the PR description and make sure the checks pass. If your change affects code that runs on GPUs, start the GPU tests by commenting `/azp run quick` and `/azp run rocm-quick` for changes to test problems, or `/azp run` for changes to shared code. Your PR starts at `priority/normal`. If it is more or less urgent than that, say so with `/set-priority <low|high>`. Add a `change/<value>` label to say why you made this change, and any other labels that apply.
3. **Mark your PR ready for review** with the **Ready for review** button.
    - **Done for you:** the code owners are asked for a review, and your labels are updated.
4. **Wait for a reviewer to respond.** There is nothing for you to do at this step.
    - **Done for you:** a reviewer takes on your PR by commenting `/accept-review <YYYY-MM-DD>`, with the date they will review it by. They are assigned to your PR, so you can see who has it.
5. **Respond to the review.** Your reviewer leaves their feedback as review threads. Answer each thread, make any changes, and resolve the thread once you have addressed it. If you decide against a suggestion, say why before you resolve it.
6. **Ask for another look** once every thread is resolved. Comment `/recompute-labels` so that your labels match what the PR now changes, then click **Re-request review** beside your reviewer's name.
    - **Done for you:** your reviewer is notified, and stays assigned to your PR. Repeat steps 5 and 6 until they approve it.
7. **Merge your approved PR** by clicking **Merge when ready**. This is your responsibility, since you know when your PR is finished. If you do not have write access, your reviewer does this for you.
    - **Done for you:** the merge queue tests your PR against the latest `development`, then merges it.

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
| `priority/<value>` | How soon your PR should be reviewed: `low`, `normal` or `high` | `normal` by default; change it with `/set-priority` |

These labels are set when your PR is opened, reopened or marked ready for review. They are not updated on every push, since a PR can take many commits before it settles. Comment `/recompute-labels` when you want them updated.

### Labels for You to Add

These labels tell your reviewer and the maintainers what cannot be worked out automatically. Add every one that applies to your PR, starting with `change/`. If you do not have permission to add labels, ask your reviewer.

| Category | What it tells them |
|---|---|
| `change/<value>` | Why you made this change: `feature`, `fix`, `performance`, `refactor` or `maintenance` |
| `problem/<value>` | What kind of problem your PR addresses |
| `status/<value>` | Something that affects whether your PR can merge |
| `platform/<value>` | A platform your PR affects |
| `agent/<value>` | An AI agent that found or wrote the work |

---

## Commands

### Commands You Can Use

| Command | What it does |
|---|---|
| `/help` | Lists the commands |
| `/recompute-labels` | Updates the scope and size labels to match your PR |
| `/set-priority <low\|normal\|high>` | Sets how soon your PR should be reviewed |
| `/set-milestone <title\|none>` | Adds your PR to a milestone, or removes it |

### Commands You Will See Reviewers Use

| Command | What it does |
|---|---|
| `/accept-review <YYYY-MM-DD>` | Takes on the review of a PR, with the date it will be reviewed by |
| `/release-review` | Gives the review back, so that another reviewer can accept it |
| `/create-milestone <title>` | Creates a new milestone that PRs can be added to |

### How Commands Respond

1. **Start your comment with the command.** A command only runs at the start of a comment. A command in a quote or in code is treated as a mention and does not run.
2. **Write the command exactly.** A misspelt command is ignored, since other tools on Quokka also use `/` commands. A command with a missing or invalid argument gets a reply showing how to use it.
3. **Look for the reaction.** A thumbs-up on your comment means the command ran. If it did not run, a reply says why.

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

2. **Once that PR has merged,** update your branch from `development` and comment `/recompute-labels`.
3. **If the file was committed by accident,** remove it and comment `/recompute-labels`.
4. **If no scope fits, or you are unsure,** reach out to the team.
