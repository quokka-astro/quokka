# Radiation-Matter Coupling: Reimplementation Design

This document specifies the module structure, data flow, and implementation strategy for the reimplementation of Quokka's radiation-matter coupling source-term update. It is the companion to `physics.md`, which defines the algorithm and equations. This document defines how that algorithm maps to code.

## Design decisions


| Decision                     | Choice                                                                              | Rationale                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Single-group / multi-group   | Unified; `nGroups_ == 1` via `constexpr if`                                         | Eliminates duplicated solver path                                                          |
| Jacobian strategy dispatch   | Compile-time for gas-only vs dust; runtime for coupled vs decoupled                 | Matches problem structure: dust model is compile-time, weak-coupling threshold is per-cell |
| Chemical bands + PE heating  | Operator-split Step A (before thermal coupling)                                     | Updated chemical-band fluxes feed directly into work term                                  |
| PE heating in thermal solve  | Absorbed into `Egas0` by Step A modifying cell state in place                       | Simpler iteration; forward-compatible with full photochemistry                             |
| `beta_order_`                | `<= 1` only; higher order deferred to future PR                                     | Simplifies flux update; multi-group never supported `>= 2`                                 |
| Isothermal (`gamma_ == 1.0`) | Early return in orchestration layer                                                 | Thermal/dust/Jacobian modules never see isothermal logic                                   |
| Debug tooling                | `constexpr bool debug_mode` template parameter; structured `DiagnosticTrace` struct | Compile-time zero-cost when off; structured output when on                                 |
| Opacity model logic          | Pushed behind physics evaluation interface                                          | Newton loop only sees `kappaP[n]`, `kappaE[n]`, `kappaF[n]`                                |
| Newton overshoot safeguard   | Line-search damping along Newton direction                                          | Maintains consistency between `Egas` and `R[n]`                                            |
| `use_D_as_base`              | Dropped; store `R[n]` directly                                                      | Per physics.md                                                                             |
| Weak-coupling fallback       | Kept; per-cell runtime dispatch                                                     | Accept warp divergence; spatially coherent in practice                                     |
| Floors during iteration      | `Erad_floor`, `Cv * tempFloor_`, `dustTempFloor_`                                   | Per physics.md                                                                             |
| File organization            | `src/radiation/`, replacing old files                                               | Directory is small; no subdirectory needed                                                 |
| Chemical band identification | `Chemistry_Traits` with `nChemicalGroups` and `chemical_band_roles` array           | Extensible to future ionization bands; compile-time role dispatch                          |


## Chemistry_Traits and group layout

### Group partitioning

The `nGroups`_ radiation groups are partitioned into thermal and chemical bands:

```
Group indices:  [0, nThermalGroups_)     [nThermalGroups_, nGroups_)
                 ├── thermal ──┤          ├── chemical ──┤
```

Thermal groups participate in the Newton solve on `(Egas, R[n])`. Chemical groups are handled by Step A (operator-split attenuation + heating) and contribute to the work term in Step B, but have no `R[n]` unknowns.

### Chemistry_Traits

```cpp
enum class ChemicalBandRole { PE, HI_ion, HeI_ion, HeII_ion };

template <typename problem_t>
struct Chemistry_Traits {
    static constexpr int nChemicalGroups = 0;
    static constexpr std::array<ChemicalBandRole, nChemicalGroups> chemical_band_roles = {};
};
```

Problems with chemical bands specialize:

```cpp
template <>
struct Chemistry_Traits<MyProblem> {
    static constexpr int nChemicalGroups = 1;
    static constexpr std::array<ChemicalBandRole, 1> chemical_band_roles = { ChemicalBandRole::PE };
};
```

### Derived compile-time quantities

Computed in `RadSystem` or a helper, from `Physics_Traits` and `Chemistry_Traits`:

```cpp
static constexpr int nChemicalGroups_ = Chemistry_Traits<problem_t>::nChemicalGroups;
static constexpr int nThermalGroups_ = nGroups_ - nChemicalGroups_;

// PE band index in the global group array, or -1 if no PE band
static constexpr int PE_group_index_ = detail::FindChemicalBand<problem_t>(ChemicalBandRole::PE);
static constexpr bool has_PE_heating_ = (PE_group_index_ >= 0);
```

where `detail::FindChemicalBand` is a `constexpr` helper:

```cpp
template <typename problem_t>
constexpr auto FindChemicalBand(ChemicalBandRole role) -> int {
    constexpr auto& roles = Chemistry_Traits<problem_t>::chemical_band_roles;
    for (int i = 0; i < Chemistry_Traits<problem_t>::nChemicalGroups; ++i) {
        if (roles[i] == role) {
            return nThermalGroups_ + i;  // offset to global group index
        }
    }
    return -1;
}
```

### Compile-time validation

```cpp
static_assert(nChemicalGroups_ >= 0 && nChemicalGroups_ <= nGroups_);
static_assert(nThermalGroups_ >= 1, "Must have at least one thermal group");
// At most one PE band:
static_assert(detail::CountChemicalBand<problem_t>(ChemicalBandRole::PE) <= 1);
// No duplicate roles:
static_assert(detail::AllUniqueRoles<problem_t>());
```

## Operator-split sequencing

The source-term update for a cell proceeds in two operator-split steps:

```
Step A: Chemical-band update (FUV, LyC, X-ray)
  - Only runs when nChemicalGroups_ > 0 (constexpr if)
  - Attenuate chemical-band Erad[n] and Frad[n] by dust opacity
  - If has_PE_heating_: compute PE heating and add to Egas
  - Modify cell state in place: Egas, Erad[chem], Frad[chem]

Step B: Thermal radiation-matter-dust coupling
  - Reads the already-modified cell state as initial condition
  - Outer work-lag iteration:
      Inner Newton solve on (Egas, R[0], ..., R[N_thermal-1])
      Flux/momentum update for all groups (thermal + chemical)
      Work-term convergence check
  - Write back: Erad[n], Frad[n], gas momentum, gas energy
```

### Step A detail

Step A runs inside the same `ParallelFor` kernel as Step B, before the thermal coupling logic. It is guarded by `if constexpr (nChemicalGroups_ > 0)`.

For each chemical group `g` in `[nThermalGroups_, nGroups_)`:

```
1. Compute dust opacity kappaF[g] at current dust temperature (or gas temperature if no dust model)
2. Attenuate radiation energy:
     Erad[g] *= exp(-rho * kappaF[g] * chat * dt)    [or implicit: Erad[g] /= (1 + rho * kappaF[g] * chat * dt)]
3. Attenuate radiation flux (same attenuation factor, component-wise):
     Frad[g] *= same_factor
4. If g == PE_group_index_:
     PE_heat = PE_heating_rate(Erad[g], rho, ...) * dt
     Egas += PE_heat
```

The attenuation uses the same implicit form as the flux relaxation in Step B (denominator `1 + rho * kappaF * chat * dt`) for consistency and stability. The PE heating uses the *attenuated* `Erad[g]`, which is the correct post-attenuation value.

After Step A completes, `Egas`, `Erad[chem]`, and `Frad[chem]` in the conserved state reflect the chemical-band update. Step B reads these modified values as its initial condition.

## File layout

All files in `src/radiation/`. Files that are unchanged are not listed.


| File                     | Responsibility                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coupling_types.hpp`     | Data structs: `CouplingState`, `ThermalResult`, `FluxResult`, `DiagnosticTrace<debug_mode>`, `Chemistry_Traits`, `ChemicalBandRole`                                                   |
| `opacity_evaluation.hpp` | `EvaluateOpacities(T_d, rho, Erad, iter) -> OpacityTerms`; all opacity-model branching lives here                                                                                     |
| `dust_closure.hpp`       | `ComputeDustTemperature`, `SelectDustModel`, gas-only specialization `Td = T`, weak-coupling fallback                                                                                 |
| `thermal_solve.hpp`      | Newton loop: residual, Jacobian (gas-only / dust-coupled / dust-decoupled), linear solve, line search, floors, convergence                                                            |
| `flux_update.hpp`        | `UpdateFluxAndMomentum`, `ComputeWorkTerm`                                                                                                                                            |
| `source_terms.hpp`       | `AddSourceTerms`: orchestration entry point. Isothermal early return, outer work-lag loop, final write-back. Replaces `source_terms_single_group.hpp`, `source_terms_multi_group.hpp` |


`radiation_dust_system.hpp` is deleted. Its Jacobian functions move into `thermal_solve.hpp`; its `SolveGasDustRadiationEnergyExchange` logic is absorbed by `thermal_solve.hpp`; its `ComputeDustTemperatureBateKeto` moves to `dust_closure.hpp`.

## Data structures

### CouplingState

Input state for one cell, assembled by the orchestration layer from the conserved variable array. Immutable after construction.

```cpp
struct CouplingState {
    double rho;
    double Egas0;                                    // gas internal energy (post-Step-A)
    double Ekin0;                                     // gas kinetic energy
    amrex::GpuArray<double, 3> gasMomentum0;
    quokka::valarray<double, nGroups_> Erad0;        // radiation energy (post-Step-A)
    quokka::valarray<double, nGroups_> Src;           // stellar radiation source * dt, scaled
    amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 3> Frad0;  // radiation flux
    amrex::GpuArray<double, nmscalars_> massScalars;
    double dt;
    double Etot0;                                     // conserved total for convergence scaling
};
```

### RadMatterContext

Immutable per-cell context for the entire radiation-matter coupling solve. Assembled once in the orchestration layer before the outer loop. Never modified during or between iterations.

```cpp
struct RadMatterContext {
    CouplingState state;              // cell initial conditions (post-Step-A)
    DustModel dust_model;             // gas_only / coupled / decoupled
    double coeff_n;                   // dust coupling coefficient (= Nd in physics.md)
    double lambda_gd_times_dt;        // precomputed gas-dust exchange for decoupled model
    double T_d0;                      // initial dust temperature
};
```

`extra_src` (= `Src + work`) is **not** part of `RadMatterContext` because `work` changes each outer iteration while the context is fixed. It is passed separately to `SolveRadiationMatterCoupling`.

### NewtonIterateState

Mutable state of the Newton iteration, updated every iteration. Passed to Jacobian and residual functions.

```cpp
struct NewtonIterateState {
    double Egas;                                      // current gas internal energy guess
    double T_gas;                                     // current gas temperature
    double T_d;                                       // current dust temperature
    quokka::valarray<double, nGroups_> Rvec;         // current R[n]
    quokka::valarray<double, nGroups_> Erad;         // current Erad[n] (recovered from R[n])
    quokka::valarray<double, nGroups_> tau;          // current optical depth: dt * rho * kappaP * chat
    OpacityTerms<problem_t> opacity_terms;            // current opacities
    double Cv;                                        // dEgas/dT at current state
};
```

### SolverParams

Solver control parameters. Not per-cell; set once from input file / compile-time constants.

```cpp
struct SolverParams {
    double resid_tol;                 // relative residual tolerance
    double rel_change_tol;            // relative change tolerance (0 to disable)
    int max_newton_iter;              // max inner Newton iterations (default 100)
    int max_outer_iter;               // max outer work-lag iterations (default 5)
};
```

### ThermalResult

Output of the inner Newton solve.

```cpp
template <typename problem_t, bool debug_mode = false>
struct ThermalResult {
    double Egas;                                      // converged gas internal energy
    double T_gas;                                     // converged gas temperature
    double T_d;                                       // converged dust temperature
    quokka::valarray<double, nGroups_> Erad;         // recovered Erad[n] from R[n]
    quokka::valarray<double, nGroups_> Rvec;         // converged R[n]
    OpacityTerms<problem_t> opacity_terms;            // final opacities (needed by flux update)
    int n_iterations;                                 // Newton iterations used
    bool converged;
    // Only present when debug_mode = true:
    DiagnosticTrace<debug_mode> trace;
};
```

### FluxResult

Output of the flux/momentum update.

```cpp
struct FluxResult {
    amrex::GpuArray<amrex::GpuArray<double, nGroups_>, 3> Frad;   // updated radiation flux
    amrex::GpuArray<double, 3> gasMomentum;                        // updated gas momentum
    quokka::valarray<double, nGroups_> Erad;                       // Erad after work-term correction
    double Egas;                                                    // Egas after work-term correction
    quokka::valarray<double, nGroups_> work;                       // recomputed work term
};
```

### DiagnosticTrace

Structured per-cell debug output, compiled away when `debug_mode = false`.

```cpp
template <bool enabled> struct DiagnosticTrace {};

template <> struct DiagnosticTrace<true> {
    static constexpr int max_recorded_iters = 20;
    int n_recorded = 0;
    struct IterationSnapshot {
        double Egas;
        double T_gas;
        double T_d;
        quokka::valarray<double, nGroups_> Rvec;
        quokka::valarray<double, nGroups_> Erad;
        double F0;                          // gas energy residual
        double Fg_abs_sum;                  // radiation energy residual norm
        double damping_factor;              // line-search alpha
    };
    amrex::GpuArray<IterationSnapshot, max_recorded_iters> snapshots;
};
```

## Module contracts

### opacity_evaluation.hpp

```cpp
/// Evaluate group-wise opacities at the given thermodynamic state.
/// All opacity-model branching (piecewise-constant, PPL fixed-slope, PPL full-spectrum)
/// is internal to this function. The caller sees only the result.
///
/// The alpha_E / alpha_P spectral exponents (for PPL full-spectrum) are internal state
/// managed by this function. They are updated when iteration_number < max_iter_to_update_alpha_E
/// and held fixed thereafter.
template <typename problem_t>
AMREX_GPU_DEVICE auto EvaluateOpacities(
    double T_d, double rho,
    quokka::valarray<double, nGroups_> const& Erad,
    int iteration_number,
    amrex::GpuArray<double, nGroups_ + 1> const& rad_boundaries,
    OpacityTerms<problem_t> const& prev_opacity  // carries alpha_E, alpha_P from previous iteration
) -> OpacityTerms<problem_t>;

/// Evaluate kappaF and delta terms. Called once at iteration 0 and again after convergence.
template <typename problem_t>
AMREX_GPU_DEVICE void EvaluateFluxOpacities(
    double T_d, double rho,
    amrex::GpuArray<double, nGroups_ + 1> const& rad_boundaries,
    quokka::valarray<double, nGroups_> const& fourPiBoverC,
    OpacityTerms<problem_t>& opacity_terms  // modified in place
);
```

### dust_closure.hpp

```cpp
enum class DustModel { gas_only, coupled, decoupled };

/// Select dust model based on coupling strength.
/// Returns gas_only when enable_dust_gas_thermal_coupling_model_ is false (constexpr).
/// Returns coupled or decoupled at runtime based on the weak-coupling threshold.
template <typename problem_t>
AMREX_GPU_DEVICE auto SelectDustModel(
    double T_gas0, double T_d0, double Egas0, double coeff_n
) -> DustModel;

/// Compute dust temperature from the current iterate.
/// - gas_only: returns T_gas
/// - coupled: Td = T_gas - sum(R) / (Nd * sqrt(T_gas))
/// - decoupled: Td from Bate-Keto balance (only at n == 0; thereafter updated by Newton step)
///
/// Enforces dustTempFloor_.
template <typename problem_t>
AMREX_GPU_DEVICE auto ComputeDustTemperature(
    DustModel model, double T_gas,
    quokka::valarray<double, nGroups_> const& Rvec,
    double coeff_n, int newton_iter,
    /* additional args for Bate-Keto when needed */
) -> double;
```

### thermal_solve.hpp

The Newton loop. This is the core of the reimplementation.

```cpp
/// Solve the thermal radiation-matter coupling for one cell.
/// ctx is immutable; extra_src changes each outer iteration.
template <typename problem_t, bool debug_mode = false>
AMREX_GPU_DEVICE auto SolveRadiationMatterCoupling(
    RadMatterContext<problem_t> const& ctx,
    quokka::valarray<double, nGroups_> const& extra_src,
    SolverParams const& params
) -> ThermalResult<problem_t, debug_mode>;
```

Pseudocode:

```
SolveRadiationMatterCoupling(ctx, extra_src, params) -> ThermalResult:

  Initialize NewtonIterateState iterate:
    iterate.Egas = ctx.state.Egas0
    iterate.Erad = ctx.state.Erad0
    iterate.Rvec = initial guess from first-iteration formula

  Newton loop (n = 0, 1, ..., params.max_newton_iter):
    iterate.T_gas = EOS::ComputeTgasFromEint(ctx.state.rho, iterate.Egas)
    iterate.T_d = ComputeDustTemperature(ctx.dust_model, iterate.T_gas, iterate.Rvec, ctx.coeff_n, n)
    fourPiBoverC = ComputeThermalRadiation(iterate.T_d)
    iterate.opacity_terms = EvaluateOpacities(iterate.T_d, ctx.state.rho, iterate.Erad, n, ...)

    if n == 0:
      EvaluateFluxOpacities(iterate.T_d, ctx.state.rho, ..., iterate.opacity_terms)
      iterate.tau = ctx.state.dt * ctx.state.rho * kappaP * chat
      iterate.Rvec = (fourPiBoverC - iterate.Erad / kappaPoverE) * iterate.tau + extra_src - ctx.state.Src
      // Note: extra_src = Src + work, so this gives R = thermal_part + work
    else:
      iterate.tau = ctx.state.dt * ctx.state.rho * kappaP * chat
      iterate.Erad[g] = kappaPoverE[g] * (fourPiBoverC[g] - (iterate.Rvec[g] - work[g]) / iterate.tau[g])
      enforce Erad floor (transfer excess to iterate.Egas)

    iterate.Cv = EOS::ComputeEintTempDerivative(ctx.state.rho, iterate.T_gas)

    Compute residual F[0], F[n] using (iterate, ctx, extra_src)
    Check convergence: |F[0]| < params.resid_tol * ctx.state.Etot0
                   AND (c/chat) * sum|F[n]| < params.resid_tol * ctx.state.Etot0
    if converged: break

    Compute Jacobian (dispatched by ctx.dust_model):
      if gas_only:   ComputeJacobianGasOnly(iterate, ctx)
      if coupled:    ComputeJacobianDustCoupled(iterate, ctx)
      if decoupled:  ComputeJacobianDustDecoupled(iterate, ctx)

    SolveArrowheadSystem(jacobian) -> (delta_x, delta_R[n])

    Line-search damping:
      alpha = 1.0
      while alpha > alpha_min:
        Egas_trial = iterate.Egas + alpha * delta_x
        R_trial = iterate.Rvec + alpha * delta_R
        T_trial = EOS::ComputeTgasFromEint(ctx.state.rho, Egas_trial)
        if T_trial > 0 and T_trial < some_bound:
          break
        alpha *= 0.5
      iterate.Egas = Egas_trial
      iterate.Rvec = R_trial

    Enforce Egas floor: iterate.Egas >= iterate.Cv * tempFloor_

    Record diagnostic snapshot if debug_mode

  Post-convergence:
    EvaluateFluxOpacities (temperature may have changed)
    Recover final Erad[n] from R[n]
    Return ThermalResult from iterate
```

#### Jacobian functions

Three static functions, all producing `JacobianResult`. The Newton loop selects which to call based on `ctx.dust_model`. Each takes `(iterate, ctx)` — the current Newton state and the immutable cell context.

**Gas-only** (`Td = T`, `dTd/dT = 1`, `dTd/dR = 0`):

```
J[0][0] = 1
J[0][n] = c / chat
J[n][0] = (1 / Cv) * X[n] * d_planck_dT
J[n][n] = -X[n] / tau[n] - 1
```

**Dust-coupled** (full `dTd/dT`, `dTd/dR` terms, Schur complement on `Fg`):

```
J[0][0] = 1
J[0][n] = c / chat
dTd_dT = 3/2 - Td / (2T)
dTd_dR = -1 / (Nd * sqrt(T))
J[n][0] = (1/Cv) * X[n] * d_planck_dTd * dTd_dT  (+ Schur terms)
J[n][n] = X[n] * d_planck_dTd * dTd_dR - X[n]/tau[n] - 1
```

Note: the Schur complement modification to `Fg` is part of this Jacobian function, not the linear solver.

**Dust-decoupled** (primary unknown is `Td`, not `Egas`):

```
J[0][0] = 0  (or d(sum R)/dTd)
J[0][n] = 1
J[n][0] = X[n] * d_planck_dTd
J[n][n] = -X[n] / tau[n] - 1
F[0] = sum(R) - lambda_gd * dt
```

Post-convergence: gas energy updated separately via implicit cooling/heating balance.

All three Jacobian functions document explicitly in inline comments that temperature derivatives of `kappaP[n]` and `kappaE[n]` are neglected.

#### Linear solver

`SolveArrowheadSystem`: Solves the "first row + first column + diagonal" system. This is the same solver for all three Jacobian variants — the structural sparsity pattern is identical. The Jacobian functions are responsible for populating the `JacobianResult` struct in the correct arrowhead form.

#### Line-search damping

Replaces the current `enable_dE_constrain` heuristic. The damping factor `alpha` is halved until:

- `T_gas(Egas_guess + alpha * delta_x) > 0`
- `T_gas(Egas_guess + alpha * delta_x) < max(T_gas_current, T_rad) * overshoot_factor`

where `overshoot_factor` is a moderate constant (e.g., 2-4). If `alpha` falls below `alpha_min` (e.g., 1e-4), accept the step anyway and rely on floor enforcement. The key improvement over the current code is that `R[n]` is always updated consistently with `Egas`: `R += alpha * delta_R` with the same `alpha`.

### flux_update.hpp

```cpp
/// Update radiation flux and gas momentum for all groups.
/// Uses the immutable cell context and converged thermal state.
/// For beta_order_ == 0: simple exponential damping.
/// For beta_order_ == 1: includes Planck and pressure terms.
template <typename problem_t>
AMREX_GPU_DEVICE auto UpdateFluxAndMomentum(
    RadMatterContext<problem_t> const& ctx,
    ThermalResult<problem_t> const& thermal,
    double gas_update_factor
) -> FluxResult;

/// Compute the work term from updated flux and momentum.
/// work[g] = (v . F[g]) * kappaF[g] * chat / c^2 * dt
template <typename problem_t>
AMREX_GPU_DEVICE auto ComputeWorkTerm(
    RadMatterContext<problem_t> const& ctx,
    FluxResult const& flux,
    OpacityTerms<problem_t> const& opacity_terms
) -> quokka::valarray<double, nGroups_>;
```

### source_terms.hpp

The orchestration entry point. Replaces `AddSourceTermsSingleGroup` and `AddSourceTermsMultiGroup`.

```
AddSourceTerms(consVar, radEnergySource, indexRange, dt, ...):

  ParallelFor(indexRange, [=] AMREX_GPU_DEVICE (i, j, k) {

    // Load cell state into CouplingState
    auto state = LoadCouplingState(consPrev, radEnergySource, i, j, k, dt);

    // Step A: Chemical-band update (operator-split, before thermal coupling)
    if constexpr (nChemicalGroups_ > 0) {
      for (int g = nThermalGroups_; g < nGroups_; ++g) {
        double kappaF_g = ComputeFluxMeanOpacity(rho, T_gas0_or_Td0, g);
        double atten = 1.0 / (1.0 + rho * kappaF_g * chat * dt);
        consVar(i,j,k, radEnergy_index + numRadVars_ * g) *= atten;
        consVar(i,j,k, x1RadFlux_index + numRadVars_ * g) *= atten;
        consVar(i,j,k, x2RadFlux_index + numRadVars_ * g) *= atten;
        consVar(i,j,k, x3RadFlux_index + numRadVars_ * g) *= atten;
      }
      if constexpr (has_PE_heating_) {
        double Erad_PE = consVar(i,j,k, radEnergy_index + numRadVars_ * PE_group_index_);
        double PE_heat = ComputePEHeating(Erad_PE, rho, ...) * dt;
        // Modify Egas in place (absorbed into Egas0 for Step B)
        consVar(i,j,k, gasInternalEnergy_index) += PE_heat;
        consVar(i,j,k, gasEnergy_index) += PE_heat;
      }
    }

    // Step B: Thermal radiation-matter-dust coupling

    // Reload cell state (now reflects Step A modifications)
    auto state = LoadCouplingState(consVar, radEnergySource, i, j, k, dt);

    // Isothermal early return
    if constexpr (gamma_ == 1.0) {
      auto ctx = RadMatterContext{ state, DustModel::gas_only, 0.0, 0.0, 0.0 };
      auto flux = UpdateFluxAndMomentumIsothermal(ctx);
      Writeback(consNew, flux, i, j, k);
      return;
    }

    // Assemble immutable per-cell context
    RadMatterContext ctx;
    ctx.state = state;
    if constexpr (enable_dust_gas_thermal_coupling_model_) {
      ctx.coeff_n = ComputeDustCouplingCoefficient(state);
      ctx.T_d0 = ComputeInitialDustTemperature(state, ctx.coeff_n);
      ctx.dust_model = SelectDustModel(T_gas0, ctx.T_d0, state.Egas0, ctx.coeff_n);
      if (ctx.dust_model == DustModel::decoupled) {
        ctx.lambda_gd_times_dt = ctx.coeff_n * sqrt(T_gas0) * (T_gas0 - ctx.T_d0);
      }
    } else {
      ctx.dust_model = DustModel::gas_only;
      ctx.coeff_n = 0.0;
    }

    // Outer work-lag iteration
    quokka::valarray<double, nGroups_> work{};
    quokka::valarray<double, nGroups_> work_prev{};

    for (int outer = 0; outer < params.max_outer_iter; ++outer) {

      // Inner Newton solve (ctx is immutable; extra_src changes each outer iteration)
      auto extra_src = ctx.state.Src + work;
      auto thermal = SolveRadiationMatterCoupling(ctx, extra_src, params);

      // Flux/momentum update
      auto flux = UpdateFluxAndMomentum(ctx, thermal, gas_update_factor);

      // Work-term convergence
      if constexpr (beta_order_ == 0 || !include_work_term_in_source) {
        Writeback(consNew, ctx, thermal, flux, gas_update_factor, i, j, k);
        break;
      }
      work_prev = work;
      work = ComputeWorkTerm(ctx, flux, thermal.opacity_terms);
      if (WorkConverged(work, work_prev, flux, ctx.state.Etot0)) {
        Writeback(consNew, ctx, thermal, flux, gas_update_factor, i, j, k);
        break;
      }
    }

    // Iteration counter updates
    ...
  });
```

## What is NOT changing

- `radiation_system.hpp`: The `RadSystem` class template, transport update, and all non-source-term functionality remain unchanged.
- `planck_integral.hpp`: Unchanged.
- Problem-specific opacity and EOS callbacks: Their signatures are unchanged. The new `EvaluateOpacities` calls the same `ComputePlanckOpacity`, `ComputeEnergyMeanOpacity`, etc.
- The hyperbolic M1 transport update: Out of scope.
- The AMReX `ParallelFor` structure and cell-centered kernel pattern: Preserved.

## Implementation order

1. `**coupling_types.hpp**`: Define all data structs and `Chemistry_Traits`. No logic beyond `constexpr` helpers for band lookup. Test: `static_assert` validation compiles for existing problems.
2. `**opacity_evaluation.hpp**`: Extract and consolidate opacity logic from `source_terms_multi_group.hpp`. Test: verify opacities match current code for all three opacity models.
3. `**dust_closure.hpp**`: Extract `ComputeDustTemperatureBateKeto`, `SelectDustModel`, add `dustTempFloor_`. Test: verify dust temperatures match current code.
4. `**thermal_solve.hpp**`: The Newton loop. Port all three Jacobian functions (gas-only, dust-coupled, dust-decoupled). Replace `enable_dE_constrain` with line-search damping. Add `DiagnosticTrace`. Test: verify converged `(Egas, R[n])` match current code for gas-only and dust cases.
5. `**flux_update.hpp**`: Extract flux/momentum update and work-term computation. Test: verify flux and momentum match current code.
6. `**source_terms.hpp**`: Orchestration. Wire Step A (chemical-band attenuation + PE heating) and Step B (thermal coupling) together. Replace `AddSourceTermsSingleGroup` and `AddSourceTermsMultiGroup`. Test: run full test suite, verify all existing tests pass.
7. **Delete old files**: `source_terms_single_group.hpp`, `source_terms_multi_group.hpp`, `radiation_dust_system.hpp`.

Each step should be independently compilable and testable against the existing test suite.