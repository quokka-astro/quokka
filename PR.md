## Summary

Fix NaN assertion failure in `phi_extended` ghost cells during `kickParticlesAllLevels()` when running with multiple GPUs and non-periodic boundary conditions (e.g., TallBoxSf test with 8 GPUs).

**Root cause:** `setFunctorParticleAccel` was a no-op, leaving ghost cells at non-periodic physical boundaries uninitialized (NaN on GPU).

**Fix:** Make `setFunctorParticleAccel` set phi ghost cells to zero at non-periodic physical boundaries, consistent with the homogeneous Dirichlet BC (`phi = 0`) used by the MLMG Poisson solver.

## How ghost cell filling works in `kickParticlesAllLevels`

For a configuration like TallBoxSf (`periodic periodic ext_dir`), `phi_extended` ghost cells are filled through a multi-step process. The key question is: which ghost cells does `setFunctorParticleAccel` actually touch?

### Level 0 (base level)

Three steps fill the ghost cells of `phi_extended`:

1. **`MultiFab::Copy(phi_extended, phi[lev], 0, 0, 1, 0)`** — copies valid (real) cells only; ghost cells are left uninitialized.

2. **`phi_extended.FillBoundary(geom[lev].periodicity())`** — handles ghost cells *inside* the domain:
   - **Interior ghost cells** (between neighboring boxes on different MPI ranks): filled by copying valid cells from the neighboring box. This is standard MPI communication.
   - **Periodic domain boundaries** (x, y): filled by copying from the opposite side of the domain via periodic wrapping.
   - **Non-periodic domain boundaries** (z): **not touched** — these ghost cells remain uninitialized.

3. **`PhysBCFunct::operator()`** (`AMReX_PhysBCFunct.H:199`) — handles ghost cells at *physical* (non-periodic) domain boundaries:
   - If all dimensions are periodic, this function **returns immediately** (line 202) and the user functor is never called.
   - Otherwise, it constructs a "grown domain" (`gdomain`) that extends the domain box in periodic dimensions only. For `periodic periodic ext_dir`, `gdomain` is grown in x and y but **not** z.
   - For each FAB, it checks whether `grow(validbox, nghost)` extends outside `gdomain`. Only FABs whose grown box protrudes beyond the z-boundary of `gdomain` trigger the fill function.
   - This means the fill function is called **only for ghost cells at z-lo and z-hi physical boundaries** — never for interior or periodic ghost cells.

   Inside the fill function (`GpuBndryFuncFab::ccfcdoit`), AMReX processes ghost cells in three passes — faces, edges, and corners — all computed relative to `gdomain`:
   - **Face boxes:** regions adjacent to `gdomain` faces. Since `gdomain` is already grown in x and y (periodic), only z-face boxes (z-lo and z-hi) can intersect the FAB's grown box.
   - **Edge boxes:** regions at the intersection of two `gdomain` faces (e.g., x-z edges, y-z edges). Since `gdomain` is grown in x and y, these edge regions are far beyond the FAB's extent and produce **empty intersections** — no ghost cells are filled here.
   - **Corner boxes:** similar to edges but at three-face intersections. Also produce **empty intersections** for the same reason.

   For each ghost cell in the non-empty boxes (z faces only), AMReX calls two functions in sequence:
   1. `FilccCell` (`AMReX_FilCC_3D_C.H`) — dispatches on the `BCRec` type. For `foextrap`, `reflect_even`, `hoextrap`, etc., it fills the ghost cell automatically. For `ext_dir`, it hits the `default: { break; }` case and **does nothing**, deferring to the user functor.
   2. `f_user` (= `setFunctorParticleAccel`) — the user-provided functor. This is where our fix acts: it now sets `dest(iv, dcomp + n) = 0.0` for each component.

### Fine levels (`lev > 0`)

`FillPatchTwoLevels` orchestrates the complete ghost cell filling for fine levels:

1. **Coarse-fine interpolation:** Creates a coarse patch covering the fine level's ghost region, fills it via `FillPatchSingleLevel` (which internally calls `FillBoundary` + `PhysBCFunct` on the coarse data), then interpolates to fine resolution. This provides ghost cell values at coarse-fine boundaries.

2. **Fine-level filling:** Calls `FillPatchSingleLevel` on the fine level with the fine-level `PhysBCFunct` (`phiBdryFunct`). This follows the same logic as level 0: `FillBoundary` handles interior and periodic ghost cells, then `PhysBCFunct` calls `setFunctorParticleAccel` **only** for ghost cells at non-periodic physical boundaries.

Both the coarse-level BC functor (`phiCoarseBdryFunct`) and fine-level BC functor (`phiBdryFunct`) use the same `setFunctorParticleAccel`, so phi is set to zero at z-boundaries at both levels.

### Summary table

| Ghost cell location | Filling mechanism | `setFunctorParticleAccel` called? |
|---|---|---|
| Interior (between boxes) | `FillBoundary` (MPI copy) | No |
| Periodic boundary (x, y) | `FillBoundary` (periodic wrapping) | No |
| Non-periodic boundary face (z) | `PhysBCFunct` → `FilccCell` (no-op for `ext_dir`) → `f_user` | **Yes** — sets phi = 0 |
| Edge/corner at periodic+non-periodic | Empty intersection with `gdomain` | No (no ghost cells to fill) |
| Coarse-fine boundary (fine levels) | `FillPatchTwoLevels` (interpolation from coarse) | No (handled by interpolation) |

## Test plan

- [x] Build TallBoxSf test successfully
- [x] Run TallBoxSf with 8 GPUs — completes all 20 timesteps without NaN assertion failure
