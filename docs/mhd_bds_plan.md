MHD BDS Reconstruction Plan
===========================

Context and current behavior
- `src/QuokkaSimulation.hpp` hard-aborts when `reconstructionOrder_ == 4` (BDS) with MHD enabled (`computeHydroFluxes` gate).
- `computeHydroFluxesBds` reconstructs only primitive gas variables with `ComputeBDSReconstructionOptimized` and then calls `ComputeFluxes` with HLLD, but never fills `leftState_bfield` / `rightState_bfield` (perpendicular B) and never computes `cc_bfield_perp_comps`. HLLD expects those arrays to carry the two transverse B components; today they stay default-initialized, so the abort protects us.
- BDS path reuses the same `faceVel` / `fast_mhd_wavespeeds` outputs needed by EMF computation, so once B-field states are present it should flow through `ComputeEMF` like other reconstruction orders.

Plan to enable BDS for MHD
1) Wire B-field data into the BDS reconstruction path
   - Build `cc_bfield_perp_comps` from `consVar_fc` via `computeCCPerpBfieldComps` (mirrors the non-BDS path) inside `computeHydroFluxesBds`.
   - Run `ComputeBDSReconstructionOptimized` on that MultiFab to populate `leftState_bfield` / `rightState_bfield` for each direction, matching the layout HLLD uses (two perpendicular components per face).
   - Keep normal B supplied directly from `consVar_fc` in `ComputeFluxes` as today.
2) Guardrails and ghost-cell requirements
   - Confirm `nghost_cc_` and `nghost_fc_` still satisfy the BDS stencil once transverse B reconstruction is added (current assert uses `bdsGhostCells + 2`; add a similar check for face-centered fields if needed).
   - Verify `nghost_vel_` (currently 3 for MHD) remains sufficient for both BDS hydro states and the EMF reconstruction steps.
3) Flux and EMF computation
   - Leave the Riemann solver choice as HLLD for BDS, but ensure `fast_mhd_wavespeeds` is passed through so `ComputeEMF` sees consistent wave-speed data.
   - Keep FO flux correction logic unchanged; it already replaces fluxes/EMFs based on `redoFlag`.
4) Clean up the BDS gate
   - Remove the MHD abort and let the BDS branch execute once transverse B states are provided. Keep the informational print and consider downgrading it to `Verbose`-guarded output if it gets noisy.

Validation strategy
- Rebuild with `reconstruction_order=4` for MHD problems and run representative tests: `ctest -R "(BrioWuShockTube|MHDBlast|FastWave)"` plus one 3D case (e.g., `MHDQuirk`).
- Compare EMF/flux outputs against PPM runs for a short integration to catch obvious stability issues; watch for NaNs in `ComputeFluxes` or EMF averages.
- If discrepancies appear, log reconstructed `leftState_bfield` slices to confirm the new BDS path honors face-centered B inputs.
