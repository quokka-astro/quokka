# DiagParticleTxt skips all output when `particles` is omitted

Severity: High

## Explanation

`DiagParticleTxt::init` documents and reports that an omitted `particles` parameter means all particle types are included. The executable diagnostic path does the opposite: when `m_particleTypes` is empty, `DiagParticleTxt::processDiag` prints a skip message and writes no particle text files.

This silently drops requested diagnostic output for the default configuration. It is especially risky because the startup message says all particle types will be included, so users can complete a run believing the text diagnostics were produced.

## Patch

```diff
diff --git a/src/io/DiagParticleTxt.H b/src/io/DiagParticleTxt.H
--- a/src/io/DiagParticleTxt.H
+++ b/src/io/DiagParticleTxt.H
@@
 	if (!m_particleTypes.empty()) {
 		// Save only specified particle types
 		particleRegister.saveParticleDataToTxtFileFiltered(plotfilename, m_particleTypes);
 	} else {
-		amrex::Print() << "DiagParticleTxt: No particle types specified, skipping output\n";
+		// Save all registered particle types
+		particleRegister.saveParticleDataToTxtFile(plotfilename);
 	}
 #endif
 }
diff --git a/src/particles/PhysicsParticles.hpp b/src/particles/PhysicsParticles.hpp
--- a/src/particles/PhysicsParticles.hpp
+++ b/src/particles/PhysicsParticles.hpp
@@
 	// Save only specified particle types to text files
 	void saveParticleDataToTxtFileFiltered(const std::string &plotfilename, const std::vector<std::string> &particleTypeNames)
 	{
 		const BL_PROFILE("PhysicsParticleRegister::saveParticleDataToTxtFileFiltered()");
@@
 		}
 	}
+
+	// Save all particle types to text files
+	void saveParticleDataToTxtFile(const std::string &plotfilename)
+	{
+		const BL_PROFILE("PhysicsParticleRegister::saveParticleDataToTxtFile()");
+		for (const auto &[type, descriptor] : particleRegistry_) {
+			descriptor->saveParticleDataToTxtFile(plotfilename, getParticleTypeName(type));
+		}
+	}
 
 	// Write all particle data to checkpoint file
 	void writeCheckpoint(const std::string &checkpointname, bool include_header) const
```
