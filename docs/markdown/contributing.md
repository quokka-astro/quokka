# Contributing Guide

Bug fixes, questions, and contributions of new features are welcome!

* Please report bugs using [GitHub Issues](https://github.com/quokka-astro/quokka/issues).
* Ask questions using [GitHub Discussions](https://github.com/quokka-astro/quokka/discussions).
* All code contributions should be done via [GitHub Pull Requests](https://github.com/quokka-astro/quokka/pulls):
    * *Continuous integration* (CI) testing is used to ensure that any new features or changes produce the right answer and follow all of the code style guidelines.
    * A *pull request* (PR) should be generated from your fork of Quokka and target the `development` branch. (See "Git workflow" below for details.)
      
        The most important guidelines are:

        * **Please ensure that your PR title and first post are descriptive, since these will be used for a squashed commit message.**
        * **Don't group together unrelated changes in a single PR. Instead, create separate PRs for each logically-related set of changes to the code.**

!!! Warning
    **Please note:** If you choose to make contributions to the code,
    you hereby grant a non-exclusive, royalty-free perpetual license
    to install, use, modify, prepare derivative works, incorporate into other computer software,
    distribute, and sublicense such enhancements or derivative works thereof, in binary and source code form.

## The `quokka` developer script

The `scripts/bash/quokka` script provides a convenient interface for configuring, building, and running problems during development. Install it by copying it to a directory on your `PATH`, e.g.:

```bash
cp scripts/bash/quokka ~/.local/bin/
```

### Commands

```
quokka config <preset> [--root <path>]
    Configure the CMake build directory for the given preset.

quokka build <preset> <problem> [<problem> ...] [-j <N>] [--root <path>]
    Compile one or more specific problem targets using ninja.

quokka build <preset> --filter <glob> [-j <N>] [--root <path>]
    Compile all problem targets matching a shell glob (e.g. `Rad*`).

quokka buildrun <preset> <problem> [<problem> ...] [-j <N>] [--fpe] [--input <file>] [--root <path>]
    Build then run one or more specific problems.

quokka buildrun <preset> --filter <pattern> [-j <N>] [--fpe] [--root <path>]
    Build matching problems then run matching tests.

quokka run <preset> [<problem> ...] [--input <file>] [-j <N>] [--fpe] [--root <path>]
    Run one or more problem executables from the tests/ directory.

quokka run <preset> [--filter <regex>] [-j <N>] [--fpe] [--root <path>]
    Run all tests, or those matching a regex via ctest -R.

quokka list [--root <path>]
    List all available problem directories.

quokka target <preset> [--root <path>]
    Show all available CMake build targets.

quokka clean [--root <path>]
    Remove plotfiles, checkpoints, and output files from tests/.
```

### Presets

| Preset     | Dimensionality | Build type |
|------------|---------------|------------|
| `1d`       | 1D            | Release    |
| `3d`       | 3D            | Release    |
| `1d-debug` | 1D            | Debug      |
| `3d-debug` | 3D            | Debug      |

### Options

| Option           | Description                                                          |
|------------------|----------------------------------------------------------------------|
| `--root <path>`  | Path to the quokka repository root (default: current directory)      |
| `--input <file>` | Input file for the executable (default: `inputs/<problem>.toml`); valid only when running exactly one `<problem>` |
| `--fpe`          | Enable floating-point exception traps (invalid, overflow, div-by-0)  |
| `--filter <pattern>` | For `run`: ctest regex via `ctest -R`; for `build`: shell glob over problem names; for `buildrun`: build via glob and run via ctest regex; exclusive with positional `<problem>` args |
| `-j <N>`         | Number of parallel jobs for ninja or ctest (default: 8)              |

`build`, `run`, and `buildrun` print final summary lines (`<name> SUCCESS|FAIL|SKIPPED`) to make tail-based status checks easy.

### Typical workflow

```bash
# 1. Configure (only needed once, or after CMakeLists changes)
quokka config 3d

# 2. Build specific problems
quokka build 3d ParticleSink
quokka build 3d RadDust RadDustMG

# 3. Or build all matching problems
quokka build 3d --filter "Rad*"

# 4. Build and run in one command
quokka buildrun 3d RadDust RadTube

# 5. Run one problem
quokka run 3d ParticleSink

# 6. Run multiple problems
quokka run 3d RadDust RadDustMG

# 7. Run with a custom input file and FPE traps enabled
quokka run 3d ParticleSink --input inputs/ParticleSink_custom.toml --fpe

# 8. Run all 3D tests (8 parallel jobs)
quokka run 3d -j 8

# 9. Run tests matching a regex pattern
quokka run 3d --filter "Particle*"

# 10. List available problems (all presets)
quokka list

# 11. Clean up test output
quokka clean
```

## Git workflow

Quokka uses [git](https://git-scm.com) for version control. If you are new to git, you can follow one of these tutorials:

* [Learn git with bitbucket](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud)
* [git - the simple guide](http://rogerdudler.github.io/git-guide/)

### Make your own fork and create a branch on it

The basic workflow is:

* Fork the main repo (or update it if you already created it).
* Implement your changes and push them on a new branch `<branch_name>` on
your fork.
* Create a Pull Request from branch `<branch_name>` on your fork to branch
`development` on the main Quokka repository.

First, let us setup your local git repo. To make your own fork of the main
(`upstream`) repository, press the fork button on the [Quokka Github page](https://github.com/BenWibking/quokka).

Then, clone your fork on your local computer. If you plan on doing a lot of Quokka development,
we recommend configuring your clone to use ssh access so you won't have to enter your Github
password every time, which you can do using these commands:

```
git clone --branch development git@github.com:<myGithubUsername>/quokka.git

# Then, navigate into your repo, add a new remote for the main Quokka repo, and fetch it:
cd Quokka
git remote add upstream git@github.com:quokka-astro/quokka.git
git remote set-url --push upstream git@github.com:<myGithubUsername>/quokka.git
git fetch upstream

# We recommend setting your development branch to track the upstream one instead of your fork:
git branch -u upstream/development
```

For instructions on setting up SSH access to your Github account on a new machine, see [here.](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)

Now you are free to play with your fork (for additional information, you can visit the
[Github fork help page](https://help.github.com/en/articles/fork-a-repo)).

!!! Note  
    You do not have to re-do the setup above every time.
    Instead, in the future, you need to update the `development` branch on your fork with
    ```
    git checkout development
    git pull
    ```

Make sure you are on the `development` branch with
```
git checkout development
```
in the Quokka directory.

Create a branch `<branch_name>` (the branch name should reflect the piece
of code you want to add, like `high_order_interpolation`) with
```
git checkout -b <branch_name>
```
and do the coding you want.
Add the files you work on to the git staging area with
```
git add <file_I_created> <and_file_I_modified>
```

### Commit & push your changes

Periodically commit your changes with
```
git commit -m "This is a 50-char description to explain my work"
```

The commit message (between quotation marks) is super important in order to
follow the developments and identify bugs.

For the moment, commits are on your local repo only. You can push them to
your fork with
```
git push -u origin <branch_name>
```

If you want to synchronize your branch with the `development` branch (this is useful
when `development` is being modified while you are working on
`<branch_name>`), you can use
```
git pull upstream development
```
and fix any conflicts that may occur.

Do not merge your branch for PR into your local `development` branch,
because it will make your local `development` branch diverge from the
matching branch in the main repository after your PR is merged.

### Submit a Pull Request

A Pull Request is the way to efficiently visualize the changes you made
and to propose your new feature/improvement/fix to the Quokka project.
Right after you push changes, a banner should appear on the Github page of
your fork, with your `<branch_name>`.

* Click on the `compare & pull request` button to prepare your PR.
* It is time to communicate your changes: write a title and a description for
your PR. People who review your PR are happy to know
  * what feature/fix you propose, and why
  * how you made it (created a new class than inherits from...)
  * and anything relevant to your PR (performance tests, images, *etc.*)
* Press `Create pull request`. Now you can navigate through your PR, which
highlights the changes you made.

!!! Note
    **Please *do not* write large Pull Requests, as they are very difficult and
    time-consuming to review. As much as possible, split them into small, targeted PRs.**
    For example, if you find typos in the documentation open a pull request that only fixes typos.
    If you want to fix a bug, make a small pull request that only fixes a bug.
    
    If you want to implement a feature and are not sure how to split it,
    just open a Discussion or Issue about your plans and request feedback from other Quokka developers.

Even before your work is ready to merge, it can be convenient to create a PR
(so you can use Github tools to visualize your changes). In this case, please
make a "draft" PR using the drop-down menu next to the "Create pull request" button.

Once your pull request is made, we will review and potentially merge it.
We recommend always creating a new branch for each pull request, as per the above instructions.
Once your pull request is merged, you can delete your local PR branch with
```
git branch -D <branch_name>
```

and you can delete the remote one on your fork with
```
git push origin --delete <branch_name>
```

Generally speaking, you want to follow the following rules:

  * Do not merge your branch for PR into your local `development` branch that tracks Quokka
    `development` branch.  Otherwise your local `development` branch will diverge from Quokka
    `development` branch.

  * Do not commit in your `development` branch that tracks Quokka `development` branch.

  * Always create a new branch based off `development` branch for each pull request, unless you are
    going to use git to fix it later.

If you have accidentally committed in `development` branch, you can fix it as follows,
```
git checkout -b new_branch
git checkout development
git reset HEAD~2  # Here 2 is the number of commits you have accidentally committed in development
git checkout .
```
After this, the local `development` should be in sync with Quokka `development` and your recent
commits have been saved in `new_branch` branch.

If for some reason your PR branch has diverged from Quokka, you can try to fix it as follows.  Before
you try it, you should back up your code in case things might go wrong.
```
git fetch upstream   # assuming upstream is the remote name for the official Quokka repo
git checkout -b xxx upstream/development  # replace xxx with whatever name you like
git branch -D development
git checkout -b development upstream/development
git checkout xxx
git merge yyy  # here yyy is your PR branch with unclean history
git rebase -i upstream/development
```
You will see something like below in your editor,
```
pick 7451d9d commit message a
pick c4c2459 commit message b
pick 6fj3g90 commit message c
```

This now requires a bit of knowledge on what those commits are, which commits have been merged,
which commits are actually new.  However, you should only see your only commits.  So it should be
easy to figure out which commits have already been merged.  Assuming the first two commits have been
merged, you can drop them by replace `pick` with `drop`:

```
drop 7451d9d commit message a
drop c4c2459 commit message b
pick 6fj3g90 commit message c
```

After saving and then exiting the editor, `git log` should show a clean history based on top of
`development` branch.  You can also do `git diff yyy..xxx` to make sure nothing new was dropped.  If
all goes well, you can submit a PR using `xxx` branch.
Don't worry, if something goes wrong during the rebase, you an always `git rebase --abort` and start over.

## Code Style Guide

### Code Guidelines

Quokka developers should adhere to the following coding guidelines:

  * Use curly braces for single statement blocks. For example:
```cpp
       for (int n=0; n<10; ++n) {
           Print() << "Like this!";
       }
```
  or
```cpp
       for (int n=0; n<10; ++n) { Print() << "Like this!"; }
```
  but not
```cpp

       for (int n=0; n<10; ++n) Print() << "Not like this.";
```
  or
```cpp
       for (int n=0; n<10; ++n)
          Print() << "Not like this.";
```

  * Member variables should be suffixed with `_`. For example:
```cpp
       amrex::Real variable_;
```

These guidelines should be adhered to in new contributions to Quokka, but
please refrain from making stylistic changes to unrelated sections of code in your PRs.

### API Documentation Using Doxygen

The Doxygen documentation is designed for advanced user-developers. It aims
to maximize the efficiency of a search-and-find style of locating information.
Doxygen style comment blocks should proceed the namespace, class, function, etc.
to be documented where appropriate. For example:
```cpp
   /**
    * \brief A one line description.
    *
    * \param[in] variable A short description of the variable.
    * \param[inout] data The value of data is read and changed.
    *
    * A longer description can be included here.
    */

    void MyFunction (int variable, MultiFab& data){
    ...
```
Additional information regarding Doxygen comment formatting can be found
in the [Doxygen Manual](https://www.doxygen.nl/manual/).
