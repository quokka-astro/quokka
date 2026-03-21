# Quokka CLI v1 Design

Date: 2026-03-21
Status: Proposed

## Summary

This document specifies a structured Unix-like CLI for Quokka development and testing on a **single host** where the worktree and buildtree are typically stored in an **NFS-backed home directory**.

The design goal is to provide a stable interface for both humans and agents while preventing:

- concurrent builds in the same worktree
- runs or tests against stale binaries
- accidental reuse of incompatible build configurations
- accidental targeting of the wrong worktree
- interactive prompts that break automation

The key design choice is:

> NFS is acceptable for source files and build artifacts, but **live coordination state must not depend on NFS**.

Therefore:

- the user-facing `quokka` command is a tiny global or per-user bootstrapper
- the actual command implementation is **worktree-local**
- worktree context is bound to the CLI executable and may be activated explicitly, similar to a virtual environment
- source and build outputs live in the normal worktree/buildtree
- lock authority lives on **host-local storage**
- the local SQLite database lives on **host-local storage**
- receipts written into the buildtree are treated as durable, reconstructable metadata

## Goals

- Provide a typed CLI with stable verbs and stable resource names.
- Reuse existing Quokka structure instead of inventing a second workflow registry.
- Fail fast with interpretable diagnostics when an artifact is stale or a resource is locked.
- Support machine-readable `--json` output with stable error codes and diagnostic identifiers.
- Provide a per-worktree activation model for interactive use.
- Allow a tiny per-user or global bootstrapper while keeping the command implementation worktree-local.
- Make the one-time bootstrapper installation requirement explicit to both users and agents.
- Keep v1 conservative and correct on a single host with NFS-backed home directories.

## Non-goals

- Cross-host coordination over shared NFS.
- Arbitrary shell task execution as a primary abstraction.
- Fine-grained target dependency analysis in v1.
- Full HPC scheduler orchestration in v1.

## Operating Assumptions

- One host is authoritative for all CLI activity.
- Stateful commands operate on the worktree bound to the invoked CLI entrypoint.
- Activation is performed by sourcing a script from the target worktree.
- Build outputs may live under the worktree on NFS.
- Runtime coordination state must live on a host-local filesystem such as `tmpfs`, `/run`, or `/tmp`.
- The CLI itself is non-interactive. Activation is done by the shell script in the worktree.

## Design Principles

- **Typed grammar, not generic tasks.** Commands operate on profiles, problems, tests, suites, and runs.
- **Global bootstrap, worktree-local implementation.** Different worktrees may carry different CLI versions.
- **Activated context over cwd inference.** The active worktree is explicit and stable within a shell session.
- **Fail-fast freshness.** `run`, `test`, and `regression` do not silently rebuild by default.
- **Host-local lock authority.** Locks are held by open file descriptors on local storage.
- **Receipts over mtimes.** Artifact validity is determined by fingerprints and receipts, not file modification time.
- **Reconstructable state.** If the host-local database is lost, it can be rebuilt by scanning receipts in the buildtree.

## Installation and Activation

The Quokka CLI has two layers:

- a tiny per-user or global `quokka` bootstrapper on `PATH`
- a worktree-local command implementation inside each worktree

Each worktree contains:

```text
<worktree>/
  bin/
    quokka
  scripts/bash/
    quokka-activate.sh
```

- `bin/quokka` is the worktree-local CLI implementation entrypoint
- `scripts/bash/quokka-activate.sh` is the shell activation script

The bootstrapper resolves the target worktree and then dispatches to that worktree's `bin/quokka`.

### Bootstrapper Requirement

The bootstrapper is a required one-time installation for normal interactive use.

This requirement must be explicit in:

- installation documentation
- onboarding documentation
- activation script error messages
- agent setup instructions

Recommended install location:

```text
$HOME/.local/bin/quokka
```

or another directory already present on `PATH`.

### Bootstrapper Expectations

The bootstrapper should be:

- tiny
- stable across worktrees
- backward compatible with older worktree-local CLIs within a reasonable support window
- easy to install or replace without touching worktree state

### Activation

Interactive activation is performed by sourcing the script from the target worktree:

```bash
source ./scripts/bash/quokka-activate.sh
source ./scripts/bash/quokka-activate.sh host-3d
```

The activation script:

- exports `QUOKKA_*` environment variables used by the bootstrapper
- modifies `PS1`
- installs a `quokka_deactivate` shell function
- verifies that `quokka` is available on `PATH`

After activation, the global `quokka` bootstrapper resolves that worktree via `QUOKKA_WORKTREE_ROOT`, regardless of the current directory.

If the bootstrapper is missing, the activation script must fail immediately with an actionable message explaining that the tiny global bootstrapper must be installed first.

### Non-activated Use

Without activation, the preferred invocation is via explicit worktree selection:

```bash
quokka -C /path/to/worktree status
quokka -C /path/to/worktree build HydroWave --profile host-3d
```

Direct invocation of the worktree-local implementation also remains valid:

```bash
/path/to/worktree/bin/quokka status
```

The `-C` form is the preferred non-interactive path for agents and scripts when activation is undesirable.

Agents should treat bootstrapper availability as an environment prerequisite and verify `command -v quokka` before relying on the `-C` workflow.

### Why Global Installation Is Acceptable

Different worktrees may carry different CLI implementations. A single global `quokka` on `PATH` is acceptable only if it remains a thin dispatcher that:

- resolves the intended worktree explicitly
- dispatches to that worktree's `bin/quokka`
- keeps almost no repository-specific logic in the bootstrap layer

This preserves version isolation while keeping the user-facing command name stable.

## CLI Grammar

### Top-level Commands

```text
quokka build [target ...] [--profile PROFILE] [--json]
quokka run PROBLEM [--input FILE] [--profile PROFILE] [--json]
quokka test [TEST] [--ctest-regex REGEX] [--profile PROFILE] [--json]
quokka regression [SUITE ...] [--profile PROFILE] [--json]
quokka tidy [changed|previous|origin|dev] [--fix] [--profile PROFILE] [--json]
quokka format [changed|previous|origin|dev|all] [--json]
quokka list problems|tests|suites|profiles [--profile PROFILE] [--json]
quokka status [--profile PROFILE] [--json]
quokka lock ls|break [--scope SCOPE] [--json]
quokka clean runs|locks|profile [--profile PROFILE] [--json]
quokka doctor [locking|runtime|profile] [--profile PROFILE] [--json]
```

The bootstrapper accepts `-C PATH` or `--worktree PATH` for every command.

### Resource Types

- `profile`: a named build configuration
- `problem`: a Quokka executable target such as `HydroWave`
- `test`: a CTest test name such as `HydroWaveFc`
- `suite`: a regression INI section such as `Sedov-GPU`

### Default Behavior

- `build` configures if needed, acquires the build lock, builds the requested targets, and writes receipts.
- `run` refuses to execute if the selected problem is missing or stale.
- `test` refuses to execute if the underlying executable is missing or stale.
- `regression` refuses to execute if the referenced target is missing or stale.
- `tidy` runs `./scripts/bash/tidy.sh` against the selected profile's build directory in the resolved worktree.
- `format` runs the repository's `clang-format` pre-commit hook against selected files in the resolved worktree.
- `status` reports locks, profile configuration, and receipt freshness.

### Optional Automation Flags

The following flags are allowed but optional in v1:

- `--build-if-needed`: if a required target is stale or missing, run `build` first
- `--reconfigure`: force CMake reconfiguration before building
- `--break-lock`: override a dead lock after validation
- `--fix`: when supported by the command, apply automated fix-it hints

These flags must be explicit. They must not be implied by default command behavior.

## Configuration File

The repository root may contain `quokka.toml`.

### Example

```toml
schema = 1

[policy]
default_profile = "host-3d"
build_lock_scope = "worktree"
run_lock_scope = "worktree"
staleness = "strict"
runtime_dir_mode = "local-only"

[profile.host-3d]
build_dir = "build/host-3d"
generator = "Ninja"
defines = { CMAKE_BUILD_TYPE = "Release", AMReX_SPACEDIM = "3" }
executor = { kind = "local" }

[profile.cuda-3d]
build_dir = "build/cuda-3d"
generator = "Ninja"
defines = { CMAKE_BUILD_TYPE = "Release", AMReX_SPACEDIM = "3", AMReX_GPU_BACKEND = "CUDA" }
executor = { kind = "docker", image = "ghcr.io/quokka-astro/quokka-linux-amd64-cuda:development" }
```

### v1 Configuration Rules

- `schema` is required and must equal `1`.
- `policy.default_profile` is required.
- Each profile must define `build_dir`.
- Each profile must define `executor.kind`.
- Arbitrary shell snippets are not part of the configuration grammar.

## Activation Model

Interactive use is centered on an activated shell, similar to a virtual environment.

### Active Context

The active context consists of:

- one worktree root
- one worktree ID
- one selected profile
- one resolved runtime directory

All stateful commands operate on the active context.

### Resolution Rules

For the global bootstrapper, the worktree root is resolved in this order:

1. explicit `-C PATH` or `--worktree PATH`
2. `QUOKKA_WORKTREE_ROOT`
3. the nearest parent of `cwd` that resolves to a Quokka worktree
4. fail with `WORKTREE_UNRESOLVED`

After worktree resolution, the bootstrapper dispatches to:

```text
<worktree_root>/bin/quokka
```

Within the worktree-local implementation, the worktree may then be re-derived from its own executable path:

```text
worktree_root = realpath(dirname(argv0) + "/..")
```

The activation environment variables guide the bootstrapper. The worktree-local CLI executable remains the final authority after dispatch.

For stateful commands, the profile is resolved in this order:

1. explicit `--profile PROFILE`
2. `QUOKKA_PROFILE`
3. `policy.default_profile`

### v1 Policy

v1 treats activation as the default interactive workflow. Outside an activated shell, users and agents should invoke the bootstrapper with `-C`.

This intentionally avoids ambiguous cwd-based behavior when multiple worktrees are open on the same host.

### Environment Variables

An activated shell exports:

```text
QUOKKA_ACTIVE=1
QUOKKA_WORKTREE_ROOT=/abs/path/to/worktree
QUOKKA_WORKTREE_ID=9f3a1c7d12ab
QUOKKA_PROFILE=host-3d
QUOKKA_RUNTIME_DIR=/tmp/quokka-1000
QUOKKA_PROMPT_PREFIX=(quokka:quokka@host-3d)
```

### Prompt Semantics

When activated, the shell prompt is prefixed with:

```text
(quokka:<worktree-name>@<profile>)
```

where `<worktree-name>` is the basename of the activated worktree root.

The full worktree path remains available in `QUOKKA_WORKTREE_ROOT`. The prompt is intentionally short while the resolved worktree remains authoritative.

### Deactivation

- the activation script must install a shell-local `quokka_deactivate` function
- `quokka_deactivate` restores the previous `PATH` and `PS1`
- `quokka_deactivate` unsets `QUOKKA_*`

## Runtime Directory Selection

The CLI selects a host-local runtime directory in this order:

1. `QUOKKA_RUNTIME_DIR`, if set and allowed
2. `$XDG_RUNTIME_DIR/quokka`
3. `/tmp/quokka-$UID`

### v1 Safety Rule

If the runtime directory resolves under the worktree, home directory, or another location likely to be NFS-backed, the CLI must fail with `RUNTIME_DIR_UNSAFE`.

## Filesystem Layout

### In the Buildtree

Each profile writes durable receipts under:

```text
<build_dir>/.quokka/
  schema.json
  profile.json
  configure-receipt.json
  artifacts/
    HydroWave.json
    RadhydroShell.json
```

These files may live on NFS because they are not the live lock authority.

### On Host-Local Storage

The runtime directory contains the mutable coordination state:

```text
<runtime_dir>/
  state.db
  locks/
    wt-<worktree_id>.build.lock
    wt-<worktree_id>.run.lock
  meta/
    wt-<worktree_id>.build.json
    wt-<worktree_id>.run.json
```

## Identifiers

### Worktree ID

`worktree_id` is derived from:

- canonical repository root path
- hostname

Recommended form:

```text
sha256(hostname + "\n" + realpath(worktree_root))[0:12]
```

This keeps lock names stable within one host while avoiding collisions between independent worktrees.

### Profile ID

`profile_id` is the profile name from `quokka.toml`.

### Artifact ID

`artifact_id` is the configured target name, such as `HydroWave`.

## Locking Model

## Lock Authority

The lock authority is:

- a host-local lock file
- held by an open file descriptor
- acquired with `fcntl`/`flock` on the local filesystem

The SQLite database is **not** the lock authority. It is only an index for diagnostics and status reporting.

### Lock Types

v1 defines two lock types:

- `build`: one active build per worktree
- `run`: one active run or test per worktree

### Scope

- `build` lock scope: `worktree`
- `run` lock scope: `worktree`

The run lock is intentionally coarse in v1 because many Quokka tests and runs use shared directories such as `tests/`.

### Acquisition Rules

- `build` fails if a `build` lock exists for the same worktree.
- `run`, `test`, and `regression` fail if a `build` lock exists for the same worktree.
- `run`, `test`, and `regression` also fail if a `run` lock exists for the same worktree.
- `tidy` fails if a `build` lock exists for the same worktree.
- `format` fails if a `build` lock exists for the same worktree.

### Lock Metadata

Each lock has a sidecar JSON metadata file.

Example:

```json
{
  "schema": 1,
  "lock_type": "build",
  "worktree_id": "9f3a1c7d12ab",
  "worktree_root": "/home/user/quokka",
  "profile": "host-3d",
  "pid": 41283,
  "boot_id": "3ed0d9d5-4d47-4ca4-8c6d-7d0a95ad9fa1",
  "hostname": "login01",
  "command": ["quokka", "build", "HydroWave", "--profile", "host-3d"],
  "started_at": "2026-03-21T13:54:21Z"
}
```

### Crash Recovery

If the process dies, the kernel releases the lock automatically. If the metadata file remains:

- `quokka status` may report it as stale metadata
- the next lock attempt checks whether the `pid` is alive and whether `boot_id` matches
- if not, the stale metadata is removed and the new process proceeds

### Manual Override

`quokka lock break` may remove stale metadata and proceed only if:

- no live lock holder is present
- the boot ID does not match, or the PID is gone

If a live holder is present, `lock break` must fail with `RESOURCE_LOCKED`.

## SQLite State Database

The SQLite database is stored at:

```text
<runtime_dir>/state.db
```

### v1 SQLite Rules

- SQLite is local-only.
- SQLite uses WAL mode.
- Loss of the database must not lose correctness.
- The CLI must be able to rebuild it by scanning buildtree receipts and lock metadata.

### Suggested Schema

```sql
CREATE TABLE schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE worktree (
  worktree_id TEXT PRIMARY KEY,
  root_path TEXT NOT NULL,
  hostname TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL
);

CREATE TABLE profile (
  worktree_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  build_dir TEXT NOT NULL,
  executor_kind TEXT NOT NULL,
  configure_fingerprint TEXT,
  last_seen_at TEXT NOT NULL,
  PRIMARY KEY (worktree_id, profile_id)
);

CREATE TABLE lock_index (
  worktree_id TEXT NOT NULL,
  lock_type TEXT NOT NULL,
  profile_id TEXT,
  pid INTEGER NOT NULL,
  boot_id TEXT NOT NULL,
  hostname TEXT NOT NULL,
  metadata_path TEXT NOT NULL,
  started_at TEXT NOT NULL,
  PRIMARY KEY (worktree_id, lock_type)
);

CREATE TABLE artifact_index (
  worktree_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL,
  receipt_path TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL,
  configure_fingerprint TEXT NOT NULL,
  built_at TEXT NOT NULL,
  PRIMARY KEY (worktree_id, profile_id, artifact_id)
);

CREATE TABLE event_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  worktree_id TEXT NOT NULL,
  profile_id TEXT,
  event_type TEXT NOT NULL,
  details_json TEXT NOT NULL
);
```

## Durable Buildtree Receipts

## Schema File

`<build_dir>/.quokka/schema.json`

```json
{
  "schema": 1,
  "kind": "quokka-buildtree-state"
}
```

## Profile Receipt

`<build_dir>/.quokka/profile.json`

```json
{
  "schema": 1,
  "profile": "host-3d",
  "worktree_root": "/home/user/quokka",
  "build_dir": "/home/user/quokka/build/host-3d",
  "generator": "Ninja",
  "executor": {
    "kind": "local"
  },
  "defines": {
    "AMReX_SPACEDIM": "3",
    "CMAKE_BUILD_TYPE": "Release"
  }
}
```

## Configure Receipt

`<build_dir>/.quokka/configure-receipt.json`

```json
{
  "schema": 1,
  "configured_at": "2026-03-21T13:55:01Z",
  "configure_fingerprint": "sha256:...",
  "cmake_version": "3.30.2",
  "compiler": {
    "cxx": "clang++",
    "cxx_version": "18.1.8"
  },
  "generator": "Ninja",
  "source_root": "/home/user/quokka",
  "build_dir": "/home/user/quokka/build/host-3d",
  "profile": "host-3d",
  "defines": {
    "AMReX_SPACEDIM": "3",
    "CMAKE_BUILD_TYPE": "Release"
  }
}
```

## Artifact Receipt

`<build_dir>/.quokka/artifacts/<target>.json`

```json
{
  "schema": 1,
  "artifact_id": "HydroWave",
  "artifact_kind": "problem",
  "profile": "host-3d",
  "binary_path": "/home/user/quokka/build/host-3d/src/problems/HydroWave/HydroWave",
  "built_at": "2026-03-21T13:58:42Z",
  "source_fingerprint": "sha256:...",
  "configure_fingerprint": "sha256:...",
  "git": {
    "head": "abc123...",
    "dirty": true,
    "submodules": {
      "extern/amrex": "def456..."
    }
  },
  "inputs": {
    "default_input": "inputs/HydroWave.toml",
    "default_working_dir": "tests"
  }
}
```

## Discovery Model

The CLI does not maintain a hand-written registry of problems.

### Profile-aware Discovery

Resources are discovered from the configured buildtree for the selected profile:

- configured targets from the CMake File API or generated build metadata
- tests from `ctest --show-only=json-v1`
- regression suites from `regression/quokka-tests.ini`

### Discovery Rules

- `list problems` shows configured executable targets for the selected profile.
- `list tests` shows configured CTest entries for the selected profile.
- `list suites` shows regression INI sections.
- if a target or test is impossible under the selected profile, it is simply absent

This behavior naturally respects profile constraints such as `AMReX_SPACEDIM` and GPU backend selection.

## Fingerprints

## Configure Fingerprint

The configure fingerprint changes when the build configuration changes.

Inputs include:

- profile ID
- build directory path
- generator
- CMake executable path and version
- executor kind
- normalized CMake defines
- relevant environment overrides such as compiler selections

## Source Fingerprint

The source fingerprint changes when the built artifact may no longer match the source tree.

v1 is conservative. It should include:

- `git rev-parse HEAD`
- submodule SHAs for build-relevant submodules
- top-level `CMakeLists.txt`
- files under `cmake/`
- files under `src/`
- the selected input file, if one is part of the command being validated
- untracked and modified files under those paths

### Important v1 Rule

Do not rely on mtimes. Compute the fingerprint from file content hashes and Git state.

## Freshness Rules

### Artifact States

Each artifact is in one of these states for a selected profile:

- `ready`: receipt exists and both fingerprints match
- `missing`: receipt or binary is absent
- `stale_source`: source fingerprint differs
- `stale_configure`: configure fingerprint differs
- `unknown`: receipt is unreadable or inconsistent

### Command Behavior

- `build`: may proceed from any state
- `run`: requires `ready`
- `test`: requires `ready` for the executable used by the test
- `regression`: requires `ready` for the suite target

### Input Resolution for `run`

`quokka run PROBLEM` resolves the input file in this order:

1. explicit `--input`
2. a CTest entry whose name exactly matches `PROBLEM`
3. `inputs/<PROBLEM>.toml`, if present
4. fail with `INPUT_REQUIRED`

## Command Semantics

## `quokka build`

### Inputs

- zero or more target names
- selected profile

### Behavior

1. Resolve worktree and selected profile.
2. Resolve or create the host-local runtime directory.
3. Acquire the `build` lock.
4. Validate or create the buildtree.
5. Reconfigure if needed.
6. Discover configured targets.
7. Build the requested targets, or all default targets if none are given.
8. Write or update receipts for all successfully built requested targets.
9. Release the `build` lock.

### Failure Modes

- unknown profile
- active build lock
- runtime directory unsafe
- CMake configure failure
- tool execution failure

## `quokka run`

### Inputs

- one problem name
- optional input file
- selected profile

### Behavior

1. Resolve the problem binary for the selected profile.
2. Refuse to run if a `build` lock exists.
3. Validate the artifact receipt and current fingerprints.
4. Acquire the `run` lock.
5. Execute the problem in the resolved working directory.
6. Release the `run` lock.

### Failure Modes

- missing binary or receipt
- stale artifact
- active build lock
- active run lock
- input required but unresolved

## `quokka test`

### Inputs

- optional exact test name or `--ctest-regex`
- selected profile

### Behavior

1. Resolve tests from `ctest --show-only=json-v1`.
2. For each selected test, determine the Quokka executable it invokes.
3. Validate freshness of each required executable.
4. Acquire the `run` lock.
5. Execute `ctest` with the requested selector.
6. Release the `run` lock.

### v1 Limitation

If a CTest entry cannot be mapped to a single Quokka executable, `test` must fail with `TEST_MAPPING_UNSUPPORTED` rather than guessing.

## `quokka regression`

### Inputs

- zero or more suite names from `regression/quokka-tests.ini`
- selected profile

### Behavior

1. Parse the selected regression suites.
2. Resolve each suite target and input file.
3. Validate freshness of the suite target.
4. Acquire the `run` lock.
5. Execute the suite using the configured executor.
6. Release the `run` lock.

### v1 Limitation

`regression` is validation-first. It does not silently invoke a build unless `--build-if-needed` is explicitly provided.

## `quokka status`

### Behavior

Report:

- resolved worktree context
- active locks
- selected profile information
- configure status
- artifact readiness for requested targets or tests

`status` never takes the build or run lock.

## `quokka tidy`

### Inputs

- optional selector from `changed`, `previous`, `origin`, or `dev`
- optional `--fix`
- selected profile

### Behavior

1. Resolve the selected profile and build directory.
2. Refuse to run if a `build` lock exists.
3. Verify that `<build_dir>/compile_commands.json` exists.
4. Execute `./scripts/bash/tidy.sh <build_dir> <selector>` from the resolved worktree root.
5. If `--fix` is present, pass `--fix` through to `tidy.sh`.

### Defaults

- selector defaults to `changed`
- build directory comes from the selected profile

### Semantics

`quokka tidy` is a typed wrapper around the existing repository script:

```text
./scripts/bash/tidy.sh <build_dir> <selector>
```

It does not perform artifact freshness validation because it consumes compile commands rather than Quokka executables. It does require that the selected profile has already been configured so that `compile_commands.json` exists.

### Failure Modes

- unknown profile
- active build lock
- profile unconfigured
- invalid tidy selector
- tool execution failure

## `quokka format`

### Inputs

- optional selector from `changed`, `previous`, `origin`, `dev`, or `all`

### Behavior

1. Resolve the current worktree root.
2. Refuse to run if a `build` lock exists.
3. Verify that `pre-commit` is available.
4. Resolve the selected file set from the current worktree.
5. Execute the repository's `clang-format` pre-commit hook from the resolved worktree root.

### Defaults

- selector defaults to `changed`

### Semantics

`quokka format` follows the repository's existing `clang-format` hook definition from `.pre-commit-config.yaml`.

It is a typed, non-interactive wrapper around:

```text
pre-commit run clang-format --files <selected files...>
```

For `all`, it instead executes:

```text
pre-commit run clang-format --all-files
```

The command does **not** call `./scripts/bash/format.sh` directly because that helper is interactive and runs the full pre-commit hook set. `quokka format` is intentionally limited to the `clang-format` hook so that its behavior is stable for automation.

### Selector Semantics

Selectors use the same Git diff bases as `tidy`:

- `changed`: files modified in the working tree relative to `HEAD`
- `previous`: files modified in the previous commit
- `origin`: files differing from `origin/<current-branch>`
- `dev`: files differing from `development`
- `all`: all files eligible for the `clang-format` pre-commit hook

### No-op Behavior

If the resolved selector produces no files, `quokka format` succeeds and reports that no files were selected.

### Failure Modes

- active build lock
- invalid format selector
- `pre-commit` not installed
- tool execution failure

## Machine-readable Output

All commands support `--json`.

### Result Envelope

```json
{
  "schema": 1,
  "ok": true,
  "command": "run",
  "profile": "host-3d",
  "resource": {
    "kind": "problem",
    "name": "HydroWave"
  },
  "diagnostic": null,
  "data": {
    "binary_path": "/home/user/quokka/build/host-3d/src/problems/HydroWave/HydroWave",
    "input": "inputs/HydroWave.toml"
  }
}
```

### Error Envelope

```json
{
  "schema": 1,
  "ok": false,
  "command": "run",
  "profile": "host-3d",
  "resource": {
    "kind": "problem",
    "name": "HydroWave"
  },
  "diagnostic": {
    "id": "STALE_ARTIFACT",
    "exit_code": 21,
    "message": "HydroWave in profile host-3d is stale and must be rebuilt before it can run.",
    "details": {
      "artifact_id": "HydroWave",
      "source_fingerprint_previous": "sha256:...",
      "source_fingerprint_current": "sha256:..."
    }
  }
}
```

## Exit Codes and Diagnostics

Exit codes are stable. Diagnostic IDs are stable.

| Exit code | Diagnostic ID | Meaning |
| --- | --- | --- |
| 0 | `OK` | Command succeeded |
| 10 | `USAGE_ERROR` | Invalid CLI usage |
| 11 | `UNKNOWN_PROFILE` | Requested profile is not defined |
| 12 | `UNKNOWN_RESOURCE` | Problem, test, or suite is unknown |
| 13 | `PROFILE_UNCONFIGURED` | Buildtree is absent or not configured |
| 14 | `INPUT_REQUIRED` | No input could be resolved |
| 15 | `TEST_MAPPING_UNSUPPORTED` | Selected test cannot be mapped safely |
| 16 | `RUNTIME_DIR_UNSAFE` | Runtime dir is on an unsafe filesystem |
| 17 | `TIDY_SELECTOR_INVALID` | Tidy selector is not one of the supported values |
| 18 | `FORMAT_SELECTOR_INVALID` | Format selector is not one of the supported values |
| 19 | `PRE_COMMIT_UNAVAILABLE` | `pre-commit` is required but not installed |
| 20 | `RESOURCE_LOCKED` | Build or run lock is active |
| 21 | `STALE_ARTIFACT` | Receipt exists but fingerprints differ |
| 22 | `MISSING_ARTIFACT` | Binary or receipt is missing |
| 23 | `CONFIGURE_DRIFT` | Profile/buildtree configuration no longer matches |
| 24 | `EXECUTOR_UNAVAILABLE` | Required executor is missing or unhealthy |
| 25 | `TOOL_FAILED` | CMake, Ninja, CTest, or executor failed |
| 26 | `STATE_CORRUPT` | Receipt or local state is unreadable or inconsistent |
| 30 | `INTERNAL_ERROR` | Unexpected CLI failure |

## Recovery Semantics

### Rebuilding Local State

If `<runtime_dir>/state.db` is missing or corrupt:

1. recreate the database
2. scan host-local lock metadata
3. scan `<build_dir>/.quokka/` receipts for known profiles
4. repopulate indexes

This recovery must not require rebuilding Quokka.

### Cleaning Receipts

`quokka clean profile --profile P` may remove:

- `<build_dir>/.quokka/artifacts/*.json`
- `<build_dir>/.quokka/configure-receipt.json`

It must not remove the buildtree unless a separate explicit command is added in a future version.

## NFS-specific Rules

### Allowed on NFS

- source tree
- build tree
- artifact receipts
- logs and output files

### Not Allowed on NFS

- live lock authority
- SQLite coordination database

### Rationale

This design avoids depending on NFS locking correctness while preserving the user's normal development layout in an NFS-backed home directory.

## Implementation Notes

### Suggested Language

Python is sufficient for v1:

- `argparse`
- `tomllib`
- `sqlite3`
- `json`
- `hashlib`
- `subprocess`
- `pathlib`

### Suggested Rollout

1. Implement `list`, `status`, and `doctor`.
2. Implement buildtree receipts and conservative freshness checks.
3. Implement host-local lock handling.
4. Implement `build`.
5. Implement `format`.
6. Implement `tidy`.
7. Implement `run` and `test`.
8. Implement `regression`.

## Future Extensions

- finer-grained per-target dependency fingerprints
- scheduler-aware executors
- narrower run locks keyed by output directory
- optional build queues
- explicit provenance export for plotfiles and checkpoints
