# quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...): computes and stores `updated` (`src/turbulence/TurbulentDriving.hpp:58`) but always returns `true` (`:98`)

## Summary
computes and stores `updated` (`src/turbulence/TurbulentDriving.hpp:58`) but always returns `true` (`:98`). The return value does not reflect whether the driving field was actually updated/applied.

## Severity
`High`

## Affected File
`src/turbulence/TurbulentDriving.hpp`

## Affected Function / Symbol
`quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:541`
- Finding tags: correctness/API

## Proposed Patch
- Return the actual `updated` flag (or equivalent apply-result) so callers can distinguish no-op timesteps from applied turbulence forcing.

## Why This Is a Bug
The function computes `updated` specifically to indicate whether the driving field was refreshed, but then always returns `true`. That makes the return value misleading and unusable for callers that want to know whether a new forcing realization was actually applied this step.

## Complete Code Patch
```diff
diff --git a/src/turbulence/TurbulentDriving.hpp b/src/turbulence/TurbulentDriving.hpp
--- a/src/turbulence/TurbulentDriving.hpp
+++ b/src/turbulence/TurbulentDriving.hpp
@@
 		amrex::Gpu::streamSynchronize();
-		return true;
+		return updated;
 	}
```
