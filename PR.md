## Radiation-matter coupling reimplementation

Replaces the fragmented radiation-matter coupling source-term implementation (4 duplicated solver paths across 3 files, ~2800 lines) with a unified, modular design (~1400 lines net addition, ~1900 lines deleted).

### Changes

- **Unified solver**: `AddSourceTermsSingleGroup` and `AddSourceTermsMultiGroup` replaced by a single `AddSourceTerms` entry point. Single-group and multi-group paths are now unified via `constexpr if`.
- **Modular structure**: 6 new files with clear separation of concerns:
  - `coupling_types.hpp` — data structures, `Chemistry_Traits`, `DustModel`, `SolverParams`, `DiagnosticTrace`
  - `opacity_evaluation.hpp` — `EvaluateOpacities`, `EvaluateFluxOpacities`; all opacity-model branching encapsulated
  - `dust_closure.hpp` — `SelectDustModel`, `ComputeDustTemperatureFromIterate`
  - `thermal_solve.hpp` — `SolveRadiationMatterCoupling`, 3 Jacobian functions, `SolveArrowheadSystem`
  - `flux_update.hpp` — `UpdateFluxAndMomentum`, `ComputeWorkTerm`, `WorkConverged`
  - `source_terms.hpp` — `AddSourceTerms`, unified orchestration with outer work-lag loop
- **Chemistry_Traits**: New trait struct for partitioning radiation bands into thermal vs. chemical (e.g. PE, HI-ionizing). Extensible to future photochemistry without touching the solver core.
- **Dust temperature uses all bands**: `ComputeDustTemperatureFromIterate` sums radiation energy over all groups (thermal + chemical) because chemical-band photons absorbed by dust heat it.
- **Dropped**: `SolveLinearEqsWithLastColumn`, `use_D_as_base`, `ComputeJacobianForGasAndDustWithPE` (PE heating will be operator-split in a follow-up PR), `beta_order_ >= 2` support.

### Files

| New | Replaces |
|---|---|
| `coupling_types.hpp` | (new) |
| `opacity_evaluation.hpp` | Internalized from `source_terms_multi_group.hpp` |
| `dust_closure.hpp` | `radiation_dust_system.hpp` |
| `thermal_solve.hpp` | `source_terms_multi_group.hpp`, `radiation_dust_system.hpp` |
| `flux_update.hpp` | `source_terms_multi_group.hpp` |
| `source_terms.hpp` | `source_terms_single_group.hpp`, `source_terms_multi_group.hpp` |

Deleted: `source_terms_single_group.hpp`, `source_terms_multi_group.hpp`, `radiation_dust_system.hpp`

Net: 1414 insertions, 1907 deletions across 11 files.

### Test results

All radiation tests pass: RadStreaming, RadMarshakAsymptotic, RadhydroShockCGS, RadhydroUniformAdvecting, RadForce, RadMarshakVaytet, RadhydroShockMultigroup, RadhydroPulseMGint, RadhydroPulseMGconst, RadTube, RadhydroBB, RadDust, RadDustMG, RadMarshakDust.

Note: `RadMarshakDustPE` is excluded pending the operator-split PE heating PR.
