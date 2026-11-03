# Building a ref for review

## Setup (once per session, from the repo root)

Install `quokka` and confirm it is not stale — see **Build & Test Commands** in `CLAUDE.md`.

```bash
# Repo root — record it; substitute literally below, never as a shell variable
git rev-parse --show-toplevel
```

Every `quokka` command below carries `--root <repo-root>` and `--source default`. Keep both.

## Choosing the preset

- **Lowest dimensionality the problem allows.** Move to `2d` / `3d` only when it demands it.
- **macOS** — append `-apple` (`1d-apple`, `2d-apple`, `3d-apple`).
- **GPU-touching change** — build with the GPU preset whenever a toolchain exists. Toolchain but no device: the compile alone is the check, runtime is CI's job.
- **Slow CPU test and a GPU device present** — run on the GPU preset.

Detect toolchain and device separately: the toolchain decides whether you can compile, the device whether you can run. `nvcc` may be off the interactive PATH.

```bash
# toolchain → GPU preset suffix
if [ -x /usr/local/cuda/bin/nvcc ] || command -v nvcc &>/dev/null; then
    echo "CUDA → suffix: -cuda"
elif command -v hipcc &>/dev/null || [ -d /opt/rocm ]; then
    echo "ROCm → suffix: -hip"
else
    echo "No GPU toolchain → skip GPU build steps"
fi

# device → can you RUN on the GPU, or only compile?
nvidia-smi -L 2>/dev/null || rocm-smi --showid 2>/dev/null || ls /dev/kfd 2>/dev/null \
    || echo "No GPU device → compile-only check"
```

## Build a ref

Base:

```bash
git fetch origin development
git checkout -B dev-<slug> origin/development
git submodule update --init --recursive
```

PR — `gh pr checkout` also resolves PRs opened from forks; `--force` resets a branch left by a prior round:

```bash
gh pr checkout NNNN --branch pr-<slug> --force
git submodule update --init --recursive
```

Then, for either — `quokka config` must run after every checkout or submodule update:

```bash
quokka config -d <preset> --delete --source default --root <repo-root> -DQUOKKA_PYTHON=OFF
quokka build -d <preset> <Target> --source default --root <repo-root>
```

For the GPU build, the same two commands with `-d <Nd>-<suffix>`.

## Running a binary directly

Run from `<repo-root>/tests`, which is gitignored, so artifacts do not dirty the tree. Both paths absolute:

```bash
cd <repo-root>/tests
<repo-root>/build/<preset>/src/problems/<Target>/<Target> \
    <repo-root>/inputs/<Target>.toml <param>=<value>
```

`quokka run` already does this.

## Reading output

Ignore AMReX performance advisories — "grid blocking factor (N) is too small for reasonable performance" is a grid-layout preference, not correctness. Mention them only when the PR is about grid decomposition.

Weigh: convergence rates, error norms, SUCCESS/FAIL lines, actual abort messages.

**Marginal failures.** An error norm within ~2× of its tolerance is not automatically the PR's fault. Compare against the base branch: near-identical norms both hovering at the threshold are tolerance noise, a pre-existing note. A norm that jumps an order of magnitude between base and PR is a regression even if it still passes.
