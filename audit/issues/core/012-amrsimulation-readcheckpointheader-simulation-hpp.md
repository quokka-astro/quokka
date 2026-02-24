# AMRSimulation::readCheckpointHeader(...): parses `istep`, `dt_`, and `tNew_` lines with unbounded `while (lis >> word) { arr[i++] = ...; }` loops (`src/simulation.hpp:4437-4440`, `:4447-4450`, `:4457-4460`) into fixed-size arrays, so malformed headers with extra tokens can overflow

## Summary
parses `istep`, `dt_`, and `tNew_` lines with unbounded `while (lis >> word) { arr[i++] = ...; }` loops (`src/simulation.hpp:4437-4440`, `:4447-4450`, `:4457-4460`) into fixed-size arrays, so malformed headers with extra tokens can overflow.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::readCheckpointHeader(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:294`
- Finding tags: robustness

## Proposed Patch
- Bound the parser loop by the destination array size, validate the exact token count, and reject malformed checkpoint headers instead of continuing to parse.

## Why This Is a Bug
The parser reads token streams into `istep`, `dt_`, and `tNew_` with unbounded `while` loops and a monotonically increasing index. A malformed checkpoint header with extra tokens can write past the end of the destination arrays/vectors, corrupting memory before any validation occurs.

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 	// read in finest_level
 	is >> finest_level;
 	GotoNextLine(is);
+	const int expected_nlevels = finest_level + 1;
@@
 	{
 		std::istringstream lis(line);
 		int i = 0;
 		while (lis >> word) {
+			if (i >= expected_nlevels || i >= static_cast<int>(istep.size())) {
+				amrex::Abort("readCheckpointHeader: malformed istep line (too many entries)");
+			}
 			istep[i++] = std::stoi(word);
 		}
+		if (i != expected_nlevels) {
+			amrex::Abort("readCheckpointHeader: malformed istep line (wrong number of entries)");
+		}
 	}
@@
 	{
 		std::istringstream lis(line);
 		int i = 0;
 		while (lis >> word) {
+			if (i >= expected_nlevels || i >= static_cast<int>(dt_.size())) {
+				amrex::Abort("readCheckpointHeader: malformed dt line (too many entries)");
+			}
 			dt_[i++] = std::stod(word);
 		}
+		if (i != expected_nlevels) {
+			amrex::Abort("readCheckpointHeader: malformed dt line (wrong number of entries)");
+		}
 	}
@@
 	{
 		std::istringstream lis(line);
 		int i = 0;
 		while (lis >> word) {
+			if (i >= expected_nlevels || i >= static_cast<int>(tNew_.size())) {
+				amrex::Abort("readCheckpointHeader: malformed t_new line (too many entries)");
+			}
 			tNew_[i++] = std::stod(word);
 		}
+		if (i != expected_nlevels) {
+			amrex::Abort("readCheckpointHeader: malformed t_new line (wrong number of entries)");
+		}
 	}
```
