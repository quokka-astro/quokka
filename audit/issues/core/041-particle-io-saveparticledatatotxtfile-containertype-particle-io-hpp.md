# particle_io::saveParticleDataToTxtFile<ContainerType>(...): gathers `particle_ids` (`src/particles/particle_IO.hpp:376`) but never writes them, and integer-component output starts at index `1` (`src/particles/particle_IO.hpp:398`) instead of `0`

## Summary
gathers `particle_ids` (`src/particles/particle_IO.hpp:376`) but never writes them, and integer-component output starts at index `1` (`src/particles/particle_IO.hpp:398`) instead of `0`. This silently drops the first user integer component (and all integer data when `NInt == 1`).

## Severity
`High`

## Affected File
`src/particles/particle_IO.hpp`

## Affected Function / Symbol
`particle_io::saveParticleDataToTxtFile<ContainerType>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:562`
- Finding tags: correctness/output

## Proposed Patch
- Write the gathered particle IDs to the output stream and start integer-component serialization at index `0` so the first user integer component is preserved.

## Why This Is a Bug
The serializer gathers particle IDs but drops them, so output rows omit a primary identifier. It also starts integer-component output at index `1`, which silently discards `idata(0)` and drops all integer data when `NInt == 1`. This is a data-loss bug in the exported diagnostic format.

## Complete Code Patch
```diff
diff --git a/src/particles/particle_IO.hpp b/src/particles/particle_IO.hpp
--- a/src/particles/particle_IO.hpp
+++ b/src/particles/particle_IO.hpp
@@
 			// Write data
 			for (size_t i = 0; i < real_data.size(); ++i) {
+				outFile << particle_ids[i] << " ";
+
 				// Write position and real components
 				for (size_t j = 0; j < real_data[i].size(); ++j) {
 					outFile << std::scientific << std::setprecision(15) << real_data[i][j] << " ";
 				}
 
 				// Write integer components
-				if constexpr (ContainerType::ParticleType::NInt > 1) {
-					for (size_t j = 1; j < int_data[i].size(); ++j) {
+				if constexpr (ContainerType::ParticleType::NInt > 0) {
+					for (size_t j = 0; j < int_data[i].size(); ++j) {
 						outFile << int_data[i][j] << " ";
 					}
 				}
```
