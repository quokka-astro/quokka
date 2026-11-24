# Writing high-quality bug report reproducers

A good bug report lets another person run the same case, see the same failure, and start debugging without first reconstructing your environment. The best reproducer is **complete**, **small**, and **reliable**:

- **Complete:** it includes the full input deck, required data files, local code changes, build options, run command, and error output.
- **Small:** it uses the fewest MPI ranks, cells, timesteps, plotfiles, and external files that still demonstrate the bug.
- **Reliable:** it fails consistently, or it clearly states how often the failure occurs and how many attempts are needed to observe it.

> **Tip**
>
> A complete small case is more useful than a large production run. If the production run needs a full cluster allocation or terabytes of data, first reduce it to the smallest case that still fails.

## Preferred format: a self-contained tarball

The preferred way to share a reproducer is to upload a single compressed tarball to the [GitHub issue](https://github.com/quokka-astro/quokka/issues). The tarball should be self-contained: after unpacking it in the `tests/` directory of a Quokka checkout, a maintainer should be able to read `README.md`, run one script, and reproduce the reported behavior.

```text
quokka-reproducer-<short-description>/
|-- README.md
|-- run.sh
|-- input/
|   `-- <problem>.toml
|-- data/
|   `-- <tables-or-small-input-files>
|-- patches/
|   |-- quokka.patch
|   `-- untracked-files/
`-- output/
    `-- failure.log
```

The files have these roles:

- `README.md`: summary of the bug, expected and observed behavior, the Quokka commit, the platform, and reduction notes.
- `run.sh`: a bash script that configures, compiles, and runs the smallest failing case.
- `input/`: the complete input deck, with no required parameters omitted.
- `data/`: every data table or auxiliary input file needed by the run. Keep these small enough to attach to the issue.
- `patches/quokka.patch`: the local code changes required to reproduce the bug. It is empty when the bug reproduces on an unmodified checkout.
- `patches/untracked-files/`: new files that `git diff` cannot represent.
- `output/failure.log`: the full terminal output from the failing run, including the first useful error message.

### Creating the reproducer

Run the helper script from your Quokka checkout:

```bash
./scripts/python/create_bug_reproducer.py <short-description> \
  --input inputs/MyProblem.toml \
  --data path/to/table.dat
```

By default it creates `tests/quokka-reproducer-<short-description>/`; `tests/` is ignored by Git. The patch is generated relative to the merge-base with `origin/development`, so it includes committed branch changes, uncommitted edits to tracked files, and edits inside submodules. The merge-base commit is recorded in `README.md`, `patches/README.md`, and `run.sh`. Untracked files, including those inside submodules, are copied to `patches/untracked-files/`; the script lists them with their sizes, so check that list for large or private files, and pass `--no-untracked` to skip them. Files matched by `.gitignore` (such as new `*.hdf5` or `*.csv` tables) are not copied; the script lists them, and any the run needs must be passed with `--data`. `--input` and `--data` paths are relative to the current directory; for a directory, its contents are copied directly into `input/` or `data/`. When exactly one input file is given, `run.sh` defaults to that deck, and to a problem with the same name if one exists.

The base commit must exist in the official repository, because maintainers fetch it from there. If you work from a fork, add the official repository as a remote (for example `upstream`), fetch it, and pass `--base-ref upstream/development`. The script warns when the base commit is not on the official `development` branch.

Then change into the new directory (`cd tests/quokka-reproducer-<short-description>`) and:

1. Fill in `README.md` and paste the failing output into `output/failure.log`.
2. Edit the settings at the top of `run.sh` (`PRESET`, `PROBLEM`, `INPUT`, `MPI_RANKS`).
3. Run `bash run.sh` to check that the reproducer fails in the same way.
4. Create the tarball with the `tar` command that the helper script prints. It excludes the `quokka-src/` worktree created by `run.sh`, which also holds the run's output.

> **Warning**
>
> Do not include build directories, large plotfile series, checkpoints, core dumps, or private machine paths unless they are essential to the bug. If a large artifact is essential, describe why and include the smallest possible version.

### What `run.sh` does

`run.sh` does not modify the files of the checkout it is unpacked in, but it registers a Git worktree in that checkout's `.git` directory, where the worktree's submodule clones are also stored. On its first run, it:

1. creates a separate Git worktree in `quokka-src/` at the recorded patch base commit,
2. runs `git submodule update --init --recursive` in that worktree, using the URLs from that commit's `.gitmodules`, so submodules match that commit (the first run clones them from the network),
3. applies `patches/quokka.patch`, copies `patches/untracked-files/`, and links `quokka-src/tests/data` to the reproducer's `data/` directory.

It then configures (unless an earlier configure finished), builds the problem with the [`quokka` developer script](contributing.md#the-quokka-developer-script), and runs the executable from `quokka-src/tests/`, like `quokka run`. The `../extern/...` and `../inputs/...` paths in standard input decks therefore work, and files in the reproducer's `data/` directory can be referred to as `data/...`.

Later runs skip the setup and reuse the existing build directory, so a developer can edit files in `quokka-src/` and re-run `bash run.sh` while debugging. To start over, delete `quokka-src/` (`rm -rf quokka-src`); the next run creates it again. After deleting a reproducer directory, run `git worktree prune` in the checkout to free the space used by its worktree and submodule clones.

If the reproducer is unpacked outside a Quokka checkout, set `QUOKKA_ROOT=/path/to/quokka`. The base commit must be present in that clone; run `git fetch origin` if `run.sh` reports that it is missing.

### Customizing `run.sh`

The generated script does not need to be general. Edit it so that another developer can run it and see the same failure. For example:

- add GPU options to the `quokka config` line, such as `-DAMReX_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=8.0`;
- export environment variables such as `CUDA_LAUNCH_BLOCKING=1`;
- set `QUOKKA_SOURCE=default` (or a file path) to load the environment file described in the [Contributing Guide](contributing.md#the-quokka-developer-script), such as module loads in `~/.config/quokka/quokka.rc`;
- replace `mpirun` with your scheduler's launcher, such as `srun -n 2 --gpus-per-task=1`;
- append runtime overrides such as `max_timesteps=20` to the run command.

## Make it as small as possible

Before filing the issue, try to reduce the reproducer while preserving the failure:

- Lower `max_timesteps` to the first failing step, or just past it.
- Reduce the domain size, AMR levels, plotfile frequency, and checkpoint frequency.
- Use one MPI rank if possible. If the bug requires MPI, use the smallest rank count that still fails.
- Prefer CPU-only Debug or ASAN builds for memory errors when the same bug appears off-GPU; see the [Debugging](debugging.md) guide.
- Remove unrelated physics options, derived variables, output fields, and analysis steps one at a time.
- Replace large tables or initial condition files with the smallest equivalent file that still triggers the issue.

> **Warning**
>
> Do not shrink the case so far that it no longer demonstrates the reported bug. If a smaller version changes the failure mode, include the smallest faithful reproducer and explain what changed during reduction.

## Filing the issue

Open a [bug report](https://github.com/quokka-astro/quokka/issues/new?template=bug_report.md) and attach the tarball. The issue template lists the information to include. Paste the first useful error message as text, not a screenshot.
