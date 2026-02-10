# Refactor Regression Testing to Standalone Script

## Summary

Refactored the Azure Pipelines regression test workflow into a standalone bash script (`run-regression-tests.sh`) that can be scheduled via crontab inside a Docker container on avatargpu, without Azure-specific dependencies or module configuration. Also adds GPU conflict detection to both the new script and the existing `azure-pipelines.yml` so that the cron job and Azure CI never contend for the single GPU simultaneously.

## Changes

### New Script: `run-regression-tests.sh`

A self-contained bash script that:
- Waits for the GPU to be free before starting (avoids conflicts with Azure CI jobs)
- Runs Quokka regression tests using the AMReX regression testing framework
- Automatically detects and classifies failures (compilation errors, crashes, OOM, disk quota, timeout)
- Creates a `status.json` file with test results and error details
- Always publishes results to GitHub Pages, even on failure
- Supports both CLI arguments and environment variables for configuration
- Uses ccache for faster compilation (required)
- Parses `webTopDir` automatically from the ini file

**Workflow:**
1. **Wait** - Poll until GPU is free and no conflicting jobs are detected
2. **Setup** - Configure ccache and validate paths
3. **Run Tests** - Execute `regtest.py --clean_testdir`
4. **Detect Errors** - Analyze logs for failure patterns
5. **Create Status** - Generate `status.json` with results
6. **Publish** - Git add/commit/push to GitHub Pages

**Configuration:**
```bash
./run-regression-tests.sh [OPTIONS]
  --ini-file PATH       # regression/quokka-tests.ini (default)
  --ccache-dir PATH     # Compiler cache location
  --source-dir PATH     # Quokka source directory
```

Supports environment variables: `REGRESSION_INI_FILE`, `CCACHE_DIR`, `REGRESSION_SOURCE_DIR`.

**Status file (`status.json`):**
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

Failure statuses: `COMPILATION_ERROR`, `CRASH`, `OUT_OF_MEMORY`, `DISK_QUOTA`, `TIMEOUT`, `UNKNOWN_FAILURE`.

### GPU Conflict Detection

Both the cron script and `azure-pipelines.yml` now wait for the GPU to be free before starting. Since both run in separate Docker containers on the same host, the check uses two layers:

| Check | Method | Reliability |
|---|---|---|
| **Primary** | `nvidia-smi --query-compute-apps` | Cross-container (queries GPU driver directly) |
| **Secondary** | `pgrep` for conflicting process | Best-effort (requires shared PID namespace) |

The secondary check is asymmetric by design:
- **Cron script**: looks for `Agent.Worker` (Azure CI worker process)
- **Azure pipeline**: looks for `regtest.py` (cron regression script)

### Updated: `.ci/azure-pipelines.yml`

Added a "Wait for GPU to be free" step before CMake configure, using the same two-layer check as the cron script.

### Migration Path

`run-regression-tests.sh` replaces `.ci/azure-pipelines-regression.yml` for containerized environments:

| | Azure Pipelines (old) | Cron script (new) |
|---|---|---|
| Trigger | Scheduled via Azure | Crontab in Docker |
| Dependencies | Azure agent, pool: avatar | None (self-contained) |
| Artifacts | Azure Pipelines + GitHub Pages | GitHub Pages only |
| GPU conflict handling | None | Wait loop (nvidia-smi) |
