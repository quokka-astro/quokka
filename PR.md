# Refactor Regression Testing to Standalone Script

## Summary

Refactored the Azure Pipelines regression test workflow into a standalone bash script (`run-regression-tests.sh`) that can run inside Docker containers without Azure-specific dependencies or module configuration.

## Changes

### New Script: `run-regression-tests.sh`

A self-contained bash script that:
- Runs Quokka regression tests using the AMReX regression testing framework
- Automatically detects and classifies failures (compilation errors, crashes, OOM, disk quota)
- Creates a `status.json` file with test results and error details
- Publishes results to GitHub Pages (always, even on failure)
- Supports both CLI arguments and environment variables for configuration
- Uses ccache for faster compilation (required)

### Key Features

**Error Detection:**
- Classifies failures into: `COMPILATION_ERROR`, `CRASH`, `OUT_OF_MEMORY`, `DISK_QUOTA`, `TIMEOUT`, `UNKNOWN_FAILURE`
- Extracts relevant error context from test logs
- Always publishes results regardless of test outcome

**Status File Format:**
```json
{
  "timestamp": "2026-02-09T14:23:45Z",
  "status": "SUCCESS",
  "exit_code": 0,
  "hostname": "container-hostname",
  "branch": "main",
  "commit": "34789d93",
  "error_details": "",
  "log_file": "regression-run.log"
}
```

**Configuration Options:**
```bash
./run-regression-tests.sh [OPTIONS]
  --ini-file PATH       # regression/quokka-tests.ini (default)
  --ccache-dir PATH     # Compiler cache location
  --source-dir PATH     # Quokka source directory
```

**Note:** The web output directory (`webTopDir`) is automatically parsed from the ini file, eliminating the need for manual configuration.

### Migration Path

This script replaces `.ci/azure-pipelines-regression.yml` for containerized environments:

**Before (Azure Pipelines):**
- Tied to Azure-specific infrastructure (pool: avatar)
- Required Azure artifacts upload
- Hardcoded paths and Azure-specific conditionals

**After (Standalone Script):**
- Runs in any Docker container
- No Azure dependencies
- Configurable via CLI args or environment variables
- Self-contained with all logic in one file

### Workflow

1. **Setup** - Configure ccache and validate paths
2. **Run Tests** - Execute `regtest.py --clean_testdir`
3. **Detect Errors** - Analyze logs for failure patterns
4. **Create Status** - Generate `status.json` with results
5. **Publish** - Git add/commit/push to GitHub Pages

### Testing

Run the script in a Docker container:
```bash
./run-regression-tests.sh
```

Or with custom configuration:
```bash
./run-regression-tests.sh \
  --ini-file custom-tests.ini \
  --ccache-dir /tmp/ccache
```

## Future Work

- Azure Pipelines workflow (`.ci/azure-pipelines-regression.yml`) can be updated to use this script
- Additional failure modes can be added to error detection as needed
- Status file can be enhanced with more metadata (test counts, timing, etc.)
