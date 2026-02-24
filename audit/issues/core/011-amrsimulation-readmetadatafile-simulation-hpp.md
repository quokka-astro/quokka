# AMRSimulation::ReadMetadataFile(...): uses `feholdexcept(...)` / `fesetenv(...)` for temporary FPE masking (`src/simulation.hpp:4164-4165`, `:4192`), but `YAML::LoadFile(...)` or subsequent parsing can throw before `fesetenv(...)` runs, leaving the process FPE environment altered

## Summary
uses `feholdexcept(...)` / `fesetenv(...)` for temporary FPE masking (`src/simulation.hpp:4164-4165`, `:4192`), but `YAML::LoadFile(...)` or subsequent parsing can throw before `fesetenv(...)` runs, leaving the process FPE environment altered.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::ReadMetadataFile(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:288`
- Finding tags: robustness

## Proposed Patch
- Wrap FPE-environment masking/restoration in an RAII guard so `fesetenv(...)` always runs on normal return and exception paths.
