# AMRSimulation::FillCoarsePatch(...): face-centered branch still constructs cell-centered boundary functors (`setBoundaryFunctor`) and never uses `setBoundaryFunctorFaceVar` / `dir` (`src/simulation.hpp:3240-3250`), so custom face-variable physical BCs are skipped on coarse interpolation fills

## Summary
face-centered branch still constructs cell-centered boundary functors (`setBoundaryFunctor`) and never uses `setBoundaryFunctorFaceVar` / `dir` (`src/simulation.hpp:3240-3250`), so custom face-variable physical BCs are skipped on coarse interpolation fills.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::FillCoarsePatch(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:261`
- Finding tags: correctness

## Proposed Patch
- In the face-centered `FillCoarsePatch` branch, construct and pass `setBoundaryFunctorFaceVar` with the correct `dir` so custom face-variable physical BC callbacks run during coarse interpolation fills.

## Why This Is a Bug
For face-centered fills, `FillCoarsePatch(...)` still constructs the cell-centered boundary functor. That means custom face-variable boundary logic (`setBoundaryFunctorFaceVar`, which depends on `dir`) is never invoked for coarse interpolation fills, so face-centered physical BCs can be silently wrong at domain boundaries.

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
-	amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(setBoundaryFunctor<problem_t>{});
-	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> finePhysicalBoundaryFunctor(geom[lev], BCs, boundaryFunctor);
-	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> coarsePhysicalBoundaryFunctor(geom[lev - 1], BCs, boundaryFunctor);
-
 	if (cen == quokka::centering::cc) {
+		amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(setBoundaryFunctor<problem_t>{});
+		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> finePhysicalBoundaryFunctor(geom[lev], BCs, boundaryFunctor);
+		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> coarsePhysicalBoundaryFunctor(geom[lev - 1], BCs, boundaryFunctor);
 		amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], coarsePhysicalBoundaryFunctor, 0,
 					     finePhysicalBoundaryFunctor, 0, refRatio(lev - 1), getAmrInterpolaterCellCentered(), BCs, 0);
 	} else if (cen == quokka::centering::fc) {
+		AMREX_ASSERT(dir != quokka::direction::na);
+		using FaceBndryFunc = amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>;
+		FaceBndryFunc boundaryFunctor_fc(setBoundaryFunctorFaceVar<problem_t>{dir});
+		amrex::PhysBCFunct<FaceBndryFunc> finePhysicalBoundaryFunctor_fc(geom[lev], BCs, boundaryFunctor_fc);
+		amrex::PhysBCFunct<FaceBndryFunc> coarsePhysicalBoundaryFunctor_fc(geom[lev - 1], BCs, boundaryFunctor_fc);
 		amrex::Interpolater *face_mapper = &amrex::face_divfree_interp;
-		amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], coarsePhysicalBoundaryFunctor, 0,
-					     finePhysicalBoundaryFunctor, 0, refRatio(lev - 1), face_mapper, BCs, 0);
+		amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], coarsePhysicalBoundaryFunctor_fc, 0,
+					     finePhysicalBoundaryFunctor_fc, 0, refRatio(lev - 1), face_mapper, BCs, 0);
 	} else {
```
