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

## Why This Is a Bug
`feholdexcept(...)` changes process-wide floating-point exception handling until `fesetenv(...)` restores it. If `YAML::LoadFile(...)` or YAML parsing throws before the explicit restore call, the function exits early and leaves the FPE environment altered for the rest of the run. That can suppress expected diagnostics or change floating-point behavior globally.

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 template <typename problem_t> void AMRSimulation<problem_t>::ReadMetadataFile(std::string const &chkfilename)
 {
-	fenv_t orig_feenv;
-	feholdexcept(&orig_feenv); // disable FPE for YAML reading
+	struct ScopedFEnvHold {
+		fenv_t env{};
+		ScopedFEnvHold() { feholdexcept(&env); }
+		~ScopedFEnvHold() { fesetenv(&env); }
+	};
+	const ScopedFEnvHold scopedFEnv{}; // disable FPE for YAML reading; restore on all exits
@@
-	fesetenv(&orig_feenv); // restore FPE
 }
```
