---
name: quokka-build
description: Use when configuring, building, running, or testing Quokka — the `quokka` CLI (config/build/buildrun/run/list/target/clean), its presets (1d/2d/3d, debug, cuda/hip), the manual CMake+ninja+ctest fallback, and the formatting and clang-tidy commands.
---

# Building and testing Quokka

The `scripts/bash/quokka` script is the recommended way to configure, build, and run tests. Make sure it exists in your `PATH`; if not, run `./scripts/bash/bootstrap.sh` once from the repo root to install it (and `quokka-pre-commit.sh`) into `~/.local/bin/`. All commands accept `--root <path>` to specify the repo root when not running from it.

The script optionally sources an environment file via `--source <file>` for commands that need the build/test environment (`config`, `build`, `buildrun`, `run`, `target`). If `--source` is omitted, no environment file is sourced.

- **Configure**: `quokka config [-d <preset>] [--delete] [--source <file>] [-D<k>=<v> ...]` — runs CMake with the selected preset (default `1d`).
- **Build one or more problems**: `quokka build [-d <preset>] <problem> [<problem> ...] [-j <N>] [--source <file>]`
- **Build matching problems**: `quokka build [-d <preset>] --filter <glob> [-j <N>]` (e.g. `'Rad*'`; quote patterns)
- **Build and run (combined)**: `quokka buildrun [-d <preset>] <problem> [<problem> ...] [-j <N>] [--fpe] [--input <file>]`
- **Build and run (filtered)**: `quokka buildrun [-d <preset>] --filter <pattern> [-j <N>]`
- **Run one or more problems**: `quokka run [-d <preset>] <problem> [<problem> ...] [--input <file>] [--fpe]` (`--input` only with one problem)
- **Run all tests**: `quokka run [-d <preset>] [-j <N>]`
- **Run matching tests**: `quokka run [-d <preset>] --filter <regex>` (quote regex/globs to avoid shell expansion)
- **List problems**: `quokka list`
- **Show targets**: `quokka target [-d <preset>]`
- **Clean test output**: `quokka clean`
- **Result summary**: `build`, `run`, and `buildrun` always print final per-target summary lines (`<name> SUCCESS|FAIL|SKIPPED`), so tooling/agents can reliably inspect outcomes by tailing the command output.

Presets: `1d`, `2d`, `3d`, `1d-debug`, `2d-debug`, `3d-debug`, `1d-hip`, `2d-hip`, `3d-hip`, `1d-cuda`, `2d-cuda`, `3d-cuda` (sets dimensionality, Release/Debug build type, and optional GPU backend). Default preset is `1d`.

**Without the script (manual):**
- **Configure**: `mkdir -p build/<preset> && cd build/<preset> && cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE=<type> -DAMReX_SPACEDIM=<N>`
- **GPU Support**: Add `-DAMReX_GPU_BACKEND=CUDA` (NVIDIA) or `-DAMReX_GPU_BACKEND=HIP` (AMD)
- **Build**: `ninja -j8 <problem>` (from build directory)
- **Run all tests**: `ctest -j8`
- **Run specific test**: `ctest -R TestName`
- **List targets** (returns 1000 targets): `cmake --build . --target help`

- **Test inputs**: Located in `inputs/` directory (`.toml` files)
- **Code formatting**: `clang-format -i file.cpp` (run from `src/` directory)
- **Static analysis**: Use `scripts/tidy.sh build changed` to run clang-tidy on modified files
- **Lint options**: `scripts/tidy.sh build [changed|previous|origin|dev] [--fix]`
