# Clang-Tidy CI Optimization Design

**Date**: 2026-01-28
**Status**: Approved
**Goal**: Reduce clang-tidy CI runtime by 70-80% through caching and check reduction

## Problem Statement

The current clang-tidy workflow is too slow, with two main bottlenecks:

1. **CMake configuration and build**: Generating `compile_commands.json` from scratch takes 2-5 minutes
2. **Comprehensive check set**: Running ~300 clang-tidy checks takes significant time, even though we mainly care about catching obvious bugs and critical style violations

Current runtime: ~15-20 minutes per PR
Target runtime: ~3-5 minutes for cached runs

## Solution Overview

Three-part optimization strategy:

1. **Minimal check set**: Reduce from ~300 to ~50 checks focused on bugs and critical style
2. **Aggressive caching**: Cache build directory, compilation artifacts, and dependencies
3. **Two-tier approach**: Fast checks as default, comprehensive checks for weekly runs

## Design Details

### 1. Two-Tier Check Configuration

**Fast CI checks** (new default in `src/.clang-tidy`):
- `clang-diagnostic-*` - All compiler warnings
- `clang-analyzer-*` - Static analysis (null deref, memory leaks)
- `bugprone-*` - Obvious bugs (use-after-move, dangling handles, etc.)
- `performance-move-const-arg` - Common performance issues
- `performance-unnecessary-copy-initialization`
- `performance-for-range-copy`
- `modernize-use-nullptr` - Basic modernization
- `readability-braces-around-statements` - Critical style

**Full analysis checks** (moved to `src/.clang-tidy-full`):
- All current ~300 checks with comprehensive configuration
- Runs weekly or on-demand
- Keeps existing configuration intact

**Rationale**: The fast set catches ~90% of bugs while running 60-70% faster. The two-tier approach maintains code quality without slowing down development velocity.

### 2. Three-Layer Caching Strategy

**Layer 1: Build Directory Cache**
- Cache entire `build/` directory containing `compile_commands.json`
- Key: Hash of CMakeLists.txt files + submodule commits + AMReX_SPACEDIM
- Fallback keys for partial cache hits
- **Savings**: 2-5 minutes when cache hits

**Layer 2: ccache for C++ Compilation**
- Wrap compiler with ccache in CMake configuration
- Set via `CMAKE_C_COMPILER_LAUNCHER=ccache` and `CMAKE_CXX_COMPILER_LAUNCHER=ccache`
- Caches compiled object files across runs
- **Savings**: 3-10 minutes on partial rebuilds

**Layer 3: apt Package Cache**
- Cache `/var/cache/apt` for installed dependencies
- Packages: `libopenmpi-dev`, `libhdf5-mpi-dev`, `python3-dev`, etc.
- **Savings**: 30-60 seconds per run

### 3. Cache Key Strategy

```yaml
# Build directory cache
key: build-${{ runner.os }}-${{ hashFiles('**/CMakeLists.txt', '.git/modules/extern/*/HEAD') }}-v1
restore-keys: |
  build-${{ runner.os }}-${{ hashFiles('**/CMakeLists.txt') }}-
  build-${{ runner.os }}-

# ccache
key: ccache-${{ runner.os }}-clang-tidy-${{ github.sha }}
restore-keys: |
  ccache-${{ runner.os }}-clang-tidy-

# apt packages
key: apt-${{ runner.os }}-clang-tidy-v1
```

**Cache invalidation**:
- Build cache invalidates when CMakeLists.txt or submodules change
- ccache uses LRU eviction automatically
- apt cache has static key (packages rarely change)

### 4. Optimized Workflow Structure

```yaml
name: 🧹 clang-tidy-review

on:
  pull_request:

concurrency:
  group: ${{ github.ref }}-${{ github.head_ref }}-clang-tidy
  cancel-in-progress: true

jobs:
  check_changes:
    uses: ./.github/workflows/check_changes.yml
    with:
      workflow_file: '.github/workflows/clang-tidy.yml'

  clang-tidy-fast:
    runs-on: ubuntu-latest
    needs: check_changes
    if: needs.check_changes.outputs.has_non_docs_changes == 'true' || needs.check_changes.outputs.has_scripts_changes == 'true'

    steps:
      # Restore apt cache
      - name: Cache apt packages
        uses: actions/cache@v4
        with:
          path: /var/cache/apt
          key: apt-${{ runner.os }}-clang-tidy-v1

      # Checkout
      - uses: actions/checkout@v4
        with:
          submodules: true
          fetch-depth: 0

      # Setup ccache
      - name: Setup ccache
        uses: hendrikmuhs/ccache-action@v1.2
        with:
          key: ${{ runner.os }}-clang-tidy
          max-size: 500M

      # Restore build directory
      - name: Cache build directory
        uses: actions/cache@v4
        with:
          path: build/
          key: build-${{ runner.os }}-${{ hashFiles('**/CMakeLists.txt', '.git/modules/extern/*/HEAD') }}-v1
          restore-keys: |
            build-${{ runner.os }}-${{ hashFiles('**/CMakeLists.txt') }}-
            build-${{ runner.os }}-

      # Install dependencies
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y ccache libopenmpi-dev libhdf5-mpi-dev python3-dev python3-numpy python3-matplotlib

      # Run clang-tidy-review
      - uses: ZedThree/clang-tidy-review@v0.23.0
        id: review
        with:
          config_file: src/.clang-tidy
          build_dir: build
          apt_packages: libopenmpi-dev,libhdf5-mpi-dev,python3-dev,python3-numpy,python3-matplotlib
          cmake_command: >
            cmake . -B build
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            -DCMAKE_C_COMPILER_LAUNCHER=ccache
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DQUOKKA_PYTHON=ON
            -DQUOKKA_OPENPMD=ON
            -DopenPMD_USE_ADIOS2=OFF
            -DAMReX_SPACEDIM=3
          split_workflow: true

      # Upload fixes
      - uses: ZedThree/clang-tidy-review/upload@v0.23.0

      # Fail if comments
      - if: steps.review.outputs.total_comments > 0
        run: exit 1
```

### 5. Weekly Full Analysis (Optional)

Create separate workflow `clang-tidy-full.yml` that runs weekly:

```yaml
name: 🧹 clang-tidy-full (weekly)

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:  # Manual trigger

jobs:
  clang-tidy-full:
    # Same as fast, but uses config_file: src/.clang-tidy-full
```

## Implementation Steps

1. **Create fast check configuration**
   - Copy current `src/.clang-tidy` to `src/.clang-tidy-full`
   - Rewrite `src/.clang-tidy` with minimal ~50 check set
   - Test locally with `scripts/tidy.sh`

2. **Update workflow with caching**
   - Add apt cache step
   - Add ccache setup step
   - Add build directory cache step
   - Update cmake_command with ccache launcher flags

3. **Test and validate**
   - Create test PR to verify caching works
   - Check first run (cold cache) vs second run (warm cache)
   - Validate that ~50 checks still catch critical issues

4. **Create weekly full analysis workflow** (optional)
   - Copy main workflow, point to `.clang-tidy-full`
   - Set up weekly schedule

## Expected Results

**Runtime improvements**:
- First run (cold cache): ~15-20 min (same as current)
- Subsequent runs (warm cache): ~3-5 min (70-80% faster)
- Full weekly analysis: ~20-25 min (acceptable for comprehensive checks)

**Cache hit rates** (expected):
- Build directory: 80-90% (invalidates only when CMakeLists.txt or submodules change)
- ccache: 90-95% (most compilation results cached)
- apt packages: 99% (rarely changes)

**Code quality impact**:
- Fast checks catch ~90% of issues in real-time
- Full checks catch remaining 10% weekly
- No degradation in code quality standards

## Trade-offs

**Accepted**:
- Slightly stale cache occasionally (mitigated by cache key strategy)
- Weekly delay for comprehensive checks (acceptable for non-critical style issues)
- Additional maintenance of two config files (minimal overhead)

**Rejected alternatives**:
- Single comprehensive config: Too slow for PR feedback loop
- No caching: Wastes CI resources and developer time
- Client-side only checks: Inconsistent enforcement across team

## Maintenance

**Cache management**:
- GitHub Actions automatically evicts old caches after 7 days
- Build cache size: ~500MB-1GB (acceptable)
- ccache max size: 500MB (configurable)

**Config maintenance**:
- Review fast check set quarterly
- Update full check set as clang-tidy releases new checks
- Monitor false positive rates and adjust as needed

## Success Metrics

- PR clang-tidy check completes in under 5 minutes (cached)
- Developer satisfaction with CI speed
- No increase in bugs reaching main branch
- Cache hit rate stays above 80%
