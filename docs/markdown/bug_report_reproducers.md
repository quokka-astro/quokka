# Writing high-quality bug report reproducers

A good bug report lets another person run the same case, see the same failure, and start debugging without first reconstructing your environment. The best reproducer is **complete**, **small**, and **reliable**:

- **Complete:** it includes the full input deck, required data files, local code changes, build options, run command, and error output.
- **Small:** it uses the fewest MPI ranks, cells, timesteps, plotfiles, and external files that still demonstrate the bug.
- **Reliable:** it fails consistently, or it clearly states how often the failure occurs and how many attempts are needed to observe it.

> **Tip**
>
> A complete small case is more useful than a large production run. If the production run needs a full cluster allocation or terabytes of data, first reduce it to the smallest case that still fails.

## Preferred format: a self-contained tarball

The preferred way to share a reproducer is to upload a single compressed tarball to the GitHub issue. The tarball should be self-contained: after unpacking it next to a clean Quokka checkout, a maintainer should be able to read `README.md`, run one script, and reproduce the reported behavior.

Use this layout when possible:

```text
quokka-reproducer-<short-description>/
|-- README.md
|-- run.sh
|-- input/
|   `-- <problem>.toml
|-- data/
|   `-- <tables-or-small-input-files>
|-- patches/
|   `-- quokka.patch
`-- output/
    |-- failure.log
    `-- expected.txt
```

The files should have these roles:

- `README.md`: one-page summary of the bug, the expected behavior, the observed behavior, the Quokka commit, and the exact platform used.
- `run.sh`: a bash script that configures, compiles, and runs the smallest failing case.
- `input/`: the complete input deck, with no required parameters omitted.
- `data/`: every data table or auxiliary input file needed by the run. Keep these files small enough to attach to the issue whenever possible.
- `patches/quokka.patch`: the local code changes required to reproduce the bug, generated with `git diff`. Omit this file only when the bug reproduces on an unmodified checkout.
- `output/failure.log`: the full terminal output from the failing run, including the first useful error message.
- `output/expected.txt`: a short note describing what result you expected instead.

Create the tarball from the parent directory:

```bash
tar -czf quokka-reproducer-<short-description>.tar.gz quokka-reproducer-<short-description>/
```

Before uploading it, test the tarball in a fresh directory if practical:

```bash
tar -xzf quokka-reproducer-<short-description>.tar.gz
cd quokka-reproducer-<short-description>
bash run.sh
```

Quokka also provides a helper script that creates this directory tree and captures code changes from your current worktree. By default, the patch is generated relative to the merge-base with `origin/development`, so it includes both committed branch changes and uncommitted tracked-file edits. Untracked files are copied separately under `patches/untracked-files/` because `git diff` cannot represent files that Git has never seen.

```bash
./scripts/python/create_bug_reproducer.py <short-description> \
  --input inputs/MyProblem.toml \
  --data path/to/table.dat
```

Review and edit the generated directory before creating the final tarball.

> **Warning**
>
> Do not include full build directories, large plotfile series, checkpoints, core dumps, or private machine paths unless they are essential to the bug. If a large artifact is essential, describe why and include the smallest possible version.

## What to include

Please provide the following items when opening a [GitHub Issue](https://github.com/quokka-astro/quokka/issues):

1. **A full input deck.**
   Include the complete `.toml` or `.in` file used for the failing run, not only the parameters that changed. If the input depends on another file, include that file too.
2. **All required data tables and auxiliary files.**
   Include opacity tables, cooling tables, restart/checkpoint dependencies, particle initial condition files, or any other non-generated data needed by the run. If a file is too large to attach, explain how it was produced and provide a smaller replacement if possible.
3. **Any code modifications.**
   If the bug appears only with local edits, provide a patch, a branch, or the exact modified files. For tarball reproducers, include the patch as `patches/quokka.patch`.
4. **Exact build information.**
   Report the Quokka commit hash, submodule state if relevant, dimensionality, build type, compiler, MPI implementation, GPU backend (`CUDA`, `HIP`, or CPU-only), GPU architecture, and important CMake options.
5. **The exact run command.**
   Include the executable name, input file path, MPI command and process count, environment variables, current working directory, and scheduler options if they matter.
6. **The observed failure.**
   Paste the first clear error message, assertion, stack trace, failing diagnostic, or incorrect numerical result. Also state what you expected to happen.
7. **A note on reliability.**
   Say whether the reproducer fails every time. For intermittent failures, report the approximate failure rate and whether settings such as `CUDA_LAUNCH_BLOCKING=1` or `HIP_LAUNCH_BLOCKING=1` change the result.

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

## Write a runnable script

Every tarball should include a short bash script that compiles and runs the failing problem from a clean checkout. This removes ambiguity about paths, presets, MPI rank counts, and environment variables.

For routine local builds, prefer the `scripts/bash/quokka` helper described in the [Contributing Guide](contributing.md#the-quokka-developer-script):

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
git rev-parse --short HEAD

./scripts/bash/quokka config -d 3d --delete \
  -DQUOKKA_PYTHON=OFF

./scripts/bash/quokka build -d 3d MyProblem

mpirun -np 1 ./build/3d/src/problems/MyProblem/MyProblem \
  ./repro/MyProblem_reproducer.toml \
  max_timesteps=20
```

If the bug requires a GPU backend, scheduler command, debug build, or specific environment variables, include them in the script:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1

./scripts/bash/quokka config -d 3d-cuda --delete \
  -DAMReX_GPU_ARCH=80

./scripts/bash/quokka build -d 3d-cuda MyProblem

srun -n 2 --gpus-per-task=1 ./build/3d-cuda/src/problems/MyProblem/MyProblem \
  ./repro/MyProblem_reproducer.toml
```

The script does not need to be general. It should be specific enough that another developer can run it, see the same failure, and then edit it while debugging.

## Suggested issue layout

You can copy this outline into the issue body:

~~~markdown
## Summary

What failed, and what should have happened instead?

## Reproducer files

- Input deck:
- Data tables or auxiliary files:
- Code modifications or branch:
- Run script:

## Build and platform

- Quokka commit:
- AMReX commit:
- Compiler and version:
- MPI implementation and version:
- Build preset or CMake command:
- GPU backend and hardware, if any:

## Run command

```bash
<exact command>
```

## Failure output

```text
<first useful error message, assertion, stack trace, or diagnostic>
```

## Reduction notes

- Smallest domain size tested:
- Smallest MPI rank count tested:
- First failing timestep:
- Does it fail every time?
~~~

## What not to submit by itself

Avoid reports that only include:

- a screenshot of an error message without the text output;
- a partial parameter list instead of the full input deck;
- a production job script that depends on private paths, unavailable modules, or large unshared data files;
- a plotfile or checkpoint without the input and command that produced it;
- a description such as "the simulation crashes" without the first clear error message.

These details can still be useful as supporting context, but they are not a reproducible bug report on their own.
