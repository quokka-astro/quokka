# Empirically motivated early feedback for `StochasticStellarPop`

## Goal and scope

Implement the empirically motivated early-feedback (EMF) model of Keller, Kruijssen & Chevance (2022) for Quokka's `StochasticStellarPop` particles. The defining numerical requirement is equation 10 of [the paper](../../stac1607.pdf): each young stellar particle supplies the finite momentum increment implied by its birth mass and its age interval during the current timestep.

The first implementation should:

- be opt-in so existing calculations are unchanged;
- inject momentum from every live `StochasticStellarPop` representation of newly formed stellar mass, including low-mass composites and sampled high-mass stars;
- use `mass_at_birth`, not current mass;
- be deterministic once particles have formed;
- coexist with, but remain independent of, the existing SN and radiation feedback paths;
- conserve vector momentum and implement the paper's thermalization of cancelled momentum;
- work on CPU and GPU and preserve Quokka's roundoff-resistant deposition workflow;
- fail explicitly rather than silently omit feedback when an active particle cannot be deposited safely on the AMR hierarchy.

This plan was prepared against branch `BenWibking/EMEF` at commit `9b1a0c976`.

## Physics contract from the paper

The paper defines the cumulative specific momentum of a stellar population of age \(t\) as

\[
P(t) = \alpha p_0 \left(\frac{t}{t_{\rm FB}}\right)^{4\alpha-1},
\qquad 0 \leq t \leq t_{\rm FB}.
\]

For a particle with birth mass \(M_i\), equation 10 gives the finite increment during a timestep:

\[
\Delta p(M_i,t,\Delta t) = \alpha p_0 M_i
\left[
\left(\frac{t+\Delta t}{t_{\rm FB}}\right)^{4\alpha-1}
-
\left(\frac{t}{t_{\rm FB}}\right)^{4\alpha-1}
\right].
\]

Here \(t\) and \(t+\Delta t\) are particle ages, not absolute simulation times. Equation 10 is already a finite increment; it must **not** be multiplied by \(\Delta t\) again.

Use the following clipped form in code:

\[
x_0 = \operatorname{clamp}\left(\frac{t_{\rm step}-t_{\rm birth}}{t_{\rm FB}},0,1\right),
\qquad
x_1 = \operatorname{clamp}\left(\frac{t_{\rm step}+\Delta t-t_{\rm birth}}{t_{\rm FB}},0,1\right),
\]

\[
\Delta p = \alpha p_0 M_{\rm birth}
\left(x_1^{4\alpha-1}-x_0^{4\alpha-1}\right).
\]

Clipping both endpoints handles particles born during the interval, timesteps that cross \(t_{\rm FB}\), and particles older than \(t_{\rm FB}\) without additional state. It also makes the schedule restart-safe: no cumulative-injected-momentum particle component is needed.

The paper's median observational parameters are

- \(p_0 = 377\ {\rm km\,s^{-1}}\);
- \(t_{\rm FB} = 3.3\ {\rm Myr}\);
- physically explored \(\alpha\) range \(0.5\) to \(1.0\), with \(\alpha=1\) used for the fiducial parameter-variation runs.

At the end of the feedback interval, the total momentum is \(\alpha p_0 M_{\rm birth}\), not generally \(p_0M_{\rm birth}\). Two useful exact cases are:

- \(\alpha=0.5\): \(4\alpha-1=1\), so injection is constant in time and totals \(0.5p_0M_{\rm birth}\);
- \(\alpha=1\): \(4\alpha-1=3\), so cumulative momentum is cubic in age and totals \(p_0M_{\rm birth}\).

The paper distributes each particle's scalar impulse over neighboring cells, requires the vector sum to vanish, and thermalizes momentum cancelled by pre-existing gas motions into the particle's host cell. It does not add ejecta mass or metals as part of EMF.

## Mapping onto `StochasticStellarPop`

### Which particles participate

Apply the age-window test to every valid `StochasticStellarPop` particle with positive `mass_at_birth`, irrespective of `StellarEvolutionStage`:

- `LowMassComposite` carries the low-mass portion of the newly formed population;
- `SNProgenitor` and `HighMassNonExploding` carry the sampled high-mass portion;
- an `SNRemnant` younger than \(t_{\rm FB}\) continues its equation-10 schedule using `mass_at_birth`.

Stage-independent injection is intentional. The empirical model is normalized to the birth mass of a stellar population and is not a per-star wind or radiation model. Gating on stage would make the integrated momentum depend on sampled stellar fates and would under-inject after an unusually early SN. `Removed` or invalid particles are excluded through the normal particle-ID validity check.

Summing `mass_at_birth` over all particles produced by one star-formation event recovers the sampled stellar birth mass. The high-mass sampling makes this fluctuate around the mass removed from gas, but it is unbiased in expectation under the existing IMF sampler. EMF should not add another stochastic draw.

### Required prerequisite: preserve birth mass when particles split

`PhysicsParticleDescriptor::splitParticles()` currently divides `mass` among child particles but copies `mass_at_birth` unchanged. EMF would therefore over-inject by the split factor after restart/refinement splitting; the existing birth-mass-derived SFH is affected by the same issue.

Before enabling EMF, update the generic split path so that, when `getMassAtBirthIndex() >= 0`, each child receives the parent birth mass divided by `splitFactor`. Add a regression showing that both total current mass and total birth mass are unchanged by splitting.

## Runtime interface

Add these parameters to `particle_types.hpp` and parse them in `particleParmParse()`:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `particles.EMF_enabled` | `false` | Enable empirical early feedback for `StochasticStellarPop`. |
| `particles.EMF_p0_kmps` | `377.0` | \(p_0\) in km s\(^{-1}\). |
| `particles.EMF_tFB_Myr` | `3.3` | Duration of the early-feedback interval in Myr. |
| `particles.EMF_alpha` | `1.0` | Self-similar expansion exponent. |

Validate finite values at startup. Require `EMF_p0_kmps >= 0`, `EMF_tFB_Myr > 0`, and `0.5 <= EMF_alpha <= 1.0`. Convert to CGS once on the host before launching device kernels:

- \(p_0\): km s\(^{-1}\) to cm s\(^{-1}\);
- \(t_{\rm FB}\): Myr to seconds.

EMF must have its own enable flag. It must not be controlled by `disable_SN_feedback` or `SN_scheme`; this permits EMF-only verification and avoids conflating two independent feedback models. Leave it disabled in all existing inputs initially. Enable it in `DiskGalaxy.toml` only in a separate, reviewable rollout after the focused tests pass.

## Code design

### 1. Isolate the equation-10 calculation

Create `src/particles/particle_early_feedback.hpp` and put a small GPU-callable pure function near the top:

```cpp
AMREX_GPU_HOST_DEVICE auto earlyFeedbackMomentumIncrement(
    amrex::Real step_time, amrex::Real dt, amrex::Real birth_time,
    amrex::Real birth_mass, amrex::Real p0, amrex::Real t_fb,
    amrex::Real alpha) noexcept -> amrex::Real;
```

The helper should:

1. return zero for invalid/non-positive birth mass, non-overlapping ages, or zero `p0`;
2. clip both dimensionless age endpoints to \([0,1]\);
3. evaluate the difference of powers directly;
4. clamp a tiny negative result caused only by floating-point roundoff to zero;
5. contain no particle-stage or mesh logic.

Keeping this function pure makes the most important physics independently testable and prevents a rate-versus-increment mistake from being hidden inside deposition code.

### 2. Add a dedicated descriptor and registry operation

Extend `PhysicsParticleDescriptorBase` with a default no-op `depositEarlyFeedback(...)`. Override it for `StochasticStellarPop` through the existing descriptor template, passing the stored birth-time and birth-mass indices. Add a registry method that dispatches only when EMF is enabled.

Do not put EMF in `ParticlePropertyUpdateTraits`: luminosity/property updates mutate particles, whereas EMF is a particle-to-mesh source operation with a stencil, ghost-zone exchange, and state fixup.

Return a small statistics object containing at least:

- number of active source particles;
- total scalar momentum requested by equation 10;
- maximum post-deposition signal speed.

Use the last quantity in the same warning path as SN deposition. The first two quantities are useful for tests and verbose diagnostics; they need not be checkpointed.

### 3. Deposit a zero-net radial impulse

Use the existing SN support radius of three cells and the same four-ghost-cell buffer requirement. Sharing the immutable stencil weights is reasonable, but avoid changing SN numerical results as part of the EMF patch.

For an off-center particle, simply multiplying a symmetric weight by the cell-center radial unit vector does not guarantee zero net vector momentum. For each particle, use a three-pass correction over the stencil (mean direction, normalization, then deposition):

1. compute the weighted mean direction \(\boldsymbol{s}=\sum_j W_j\hat{\boldsymbol{r}}_j\);
2. form corrected vectors \(\boldsymbol{q}_j=W_j(\hat{\boldsymbol{r}}_j-\boldsymbol{s})\), so \(\sum_j\boldsymbol{q}_j=0\) when \(\sum_jW_j=1\);
3. renormalize by \(\sum_j|\boldsymbol{q}_j|\);
4. deposit \(\Delta\boldsymbol{p}_j = \Delta p\,\boldsymbol{q}_j/\sum_k|\boldsymbol{q}_k|\).

This enforces both paper invariants to roundoff:

\[
\sum_j\Delta\boldsymbol{p}_j = 0,
\qquad
\sum_j|\Delta\boldsymbol{p}_j| = \Delta p.
\]

Use `ProbLoArray()`, `CellSizeArray()`, and `InvCellSizeArray()` values in GPU lambdas. Do not capture `Geometry` raw pointers.

Deposit into a temporary `MultiFab`, call `SumBoundary()`, apply `ParticleUtils::roundoffMultiFab()`, and only then update the state. Unlike the SN path, EMF changes no density, so its state-application routine must gate on an explicit deposition count or momentum/energy marker rather than `d_rho != 0`.

### 4. Account for energy and cancellation explicitly

EMF constrains momentum, not a separate bolometric energy budget. The state update must nevertheless keep total and internal energy consistent when momentum changes.

Recommended implementation:

1. accumulate all momentum increments in the buffer;
2. when applying the aggregated buffer, add the exact kinetic-energy difference associated with the net cell momentum change to total gas energy, leaving recipient-cell internal energy unchanged;
3. for each particle event, compute the work \(\sum_j \boldsymbol{v}_j\cdot\Delta\boldsymbol{p}_j\) using the pre-feedback gas state;
4. if that work is negative, add its magnitude as thermal energy to both total and internal energy in the particle's host cell.

The event-wise work sum is invariant under a uniform boost because the corrected stencil has zero net vector impulse. This implements the paper's stated host-cell thermalization when existing motions cancel the injected momentum, while the aggregated kinetic-energy update prevents colocated `StochasticStellarPop` subparticles from producing an energy deficit through independently buffered kicks.

Before merging, exercise two simultaneous particles whose stencils overlap. Confirm that the chosen buffering convention cannot produce negative internal energy and document how cancellation between separate particle events is treated. If exact paper parity for inter-event cancellation cannot be established from the paper alone, state the convention in `particles.md` rather than hiding it in code.

Do not:

- inject mass, metals, passive scalars, or radiation energy;
- reuse the SN blast energy or terminal-momentum limiter;
- cap equation-10 momentum based on an assumed energy budget;
- homogenize the background gas velocity as the current optional SN path can do.

### 5. Place the operation in the timestep deliberately

In `AMRSimulation::particleMeshInteraction()`, call EMF after `createParticlesFromState()` and before `depositSN()`:

1. hydro has advanced the gas;
2. existing particles have drifted to their end-of-step positions;
3. new stellar particles have been created with the current star-formation birth-time convention;
4. EMF deposits the clipped equation-10 increment for `[time, time + dt]`;
5. SN feedback then processes particles whose death time lies in the step.

This gives newborn particles their first finite interval of feedback immediately and uses birth mass even if an SN later changes current mass. It is a first-order operator split: the interval's feedback is deposited at the particle's end-of-step position. Document that choice.

### 6. Treat AMR as an explicit support boundary

The current particle-mesh interaction deposits SN feedback only at `finest_level` and contains a TODO for AMR subcycling. EMF must not silently inherit this limitation because runaway high-mass particles can move substantially during 3.3 Myr.

Implement in two stages:

1. **Correct single-level/no-subcycling implementation.** This is the focused physics milestone and covers `TallBoxSf`-style runs. Add a runtime check that every active EMF particle is present on the level being deposited and its full three-cell stencil is addressable through valid cells plus allocated ghosts.
2. **Multilevel production support before enabling `DiskGalaxy`.** Dispatch on every particle-owning level after level synchronization (`do_subcycle = 0`), and define how a stencil crossing a coarse-fine interface is represented. The preferred solution is to keep the full active-source stencil on one level using age-gated refinement/tagging around particles younger than `tFB`; otherwise the source buffer needs conservative prolongation/restriction across the interface. Add an AMR test in which a young particle approaches a refinement boundary. An abort with a precise unsupported-configuration message is acceptable in the first milestone; omitted momentum is not.

General AMR subcycling is out of scope for the first patch and should remain explicitly rejected while EMF is enabled.

## File-level implementation map

| File | Planned change |
| --- | --- |
| `src/particles/particle_types.hpp` | Add EMF parameters, parsing, validation, and unit conversion inputs. |
| `src/particles/particle_early_feedback.hpp` | Add equation-10 helper, corrected radial stencil deposition, buffer exchange, energy application, and statistics. |
| `src/particles/PhysicsParticles.hpp` | Add descriptor/registry dispatch; pass `birth_time` and `mass_at_birth`; split `mass_at_birth` correctly when particles split. |
| `src/simulation.hpp` | Invoke EMF after particle creation and before SN; add signal-speed warning and AMR support checks. |
| `docs/markdown/particles.md` | Document the physics, equation, parameters, operator order, energy convention, and AMR boundary. |
| `src/problems/ParticleEarlyFeedback/` | Add a focused 3D CTest problem and CMake registration. |
| `inputs/ParticleEarlyFeedback.toml` | Provide the deterministic uniform-medium test input. |
| `src/problems/ParticleSF/testParticleSF.cpp` | Extend restart/splitting checks to assert conservation of total `mass_at_birth` and EMF normalization. |
| `inputs/DiskGalaxy.toml` | Opt in only after multilevel validation, in a separate change. |

No new particle real component is required. Existing checkpoints already carry `birth_time` and `mass_at_birth` in this checkout.

## Focused test plan

Create a small `ParticleEarlyFeedback` problem with an ideal-gas uniform medium and one programmatically initialized `StochasticStellarPop` particle. Disable particle drift and SN feedback unless a test specifically exercises their interaction.

### Equation-10 tests

- Before birth: zero momentum.
- At birth: the first interval matches the analytic formula.
- Crossing `tFB`: only the interval up to `tFB` is injected.
- After `tFB`: zero additional momentum.
- `alpha = 0.5`: cumulative momentum is linear and ends at `0.5 p0 M_birth`.
- `alpha = 1.0`: cumulative momentum is cubic and ends at `p0 M_birth`.
- Timestep partition: one large interval equals the sum of multiple sub-intervals at fixed particle position.
- Current mass differs from birth mass: the result continues to scale with `mass_at_birth`.

### Deposition tests

- Centered source in stationary gas:
  - integrated scalar impulse equals equation 10;
  - all three components of net vector momentum are zero to floating-point tolerance;
  - density and passive scalars are unchanged;
  - internal and total energy remain consistent.
- Off-center source: repeat the scalar- and vector-momentum checks; this is the regression for the zero-sum correction.
- Uniformly boosted medium: compare in the comoving frame and verify Galilean-invariant momentum and thermalization.
- Two colocated source particles versus one particle with their summed birth mass: momentum fields agree to roundoff.
- Two overlapping, non-colocated sources: internal energy stays positive and the documented cancellation-energy convention is satisfied.
- Feature disabled: state remains unchanged and all existing particle/SN defaults are preserved.

### Lifecycle, restart, and AMR tests

- A progenitor that becomes an SN remnant before `tFB` retains the correct remaining EMF schedule through `mass_at_birth`.
- A checkpoint/restart inside the feedback interval matches an uninterrupted run without storing cumulative EMF state.
- Restart-time particle splitting preserves total `mass_at_birth` and total equation-10 momentum.
- An active particle at a refinement boundary either deposits conservatively according to the implemented multilevel policy or produces the intentional first-milestone abort; it must never be silently skipped.
- Run with CPU plus at least one GPU backend when available. The test should use tolerances scaled to total injected scalar momentum because enabled FMA can prevent bitwise directional symmetry.

## Implementation sequence

1. **Birth-mass invariant:** fix split handling and add the `ParticleSF` restart regression.
2. **Physics kernel:** add parameters and the pure clipped equation-10 helper with analytic tests.
3. **Single-level deposition:** implement corrected radial momentum, energy bookkeeping, buffer exchange, and focused conservation/Galilean tests.
4. **Lifecycle integration:** add registry dispatch and timestep ordering; test newborns, SN overlap, and restart continuity.
5. **AMR support:** implement active-source level handling and the refinement-boundary regression.
6. **Documentation and rollout:** update `particles.md`; only then enable EMF in a production input in a separate commit.

Each slice should remain buildable and independently reviewable. Do not mix changes to the existing SN scheme's numerical results into these commits.

## Validation commands

Use the repository CLI from the project root:

```sh
quokka build -d 3d ParticleEarlyFeedback ParticleSF
quokka run -d 3d --filter 'ParticleEarlyFeedback|ParticleSF'
scripts/tidy.sh build changed
git diff --check
```

For GPU validation, repeat the focused build and tests with the relevant `3d-cuda` or `3d-hip` preset and record the exact command and backend.

## Completion criteria

The implementation is complete when:

- the timestep-integrated scalar impulse matches the clipped equation-10 schedule for all tested age intervals and both endpoint alpha cases;
- the stencil conserves vector momentum for centered and off-center particles;
- birth-mass splitting, restart continuity, and SN-remnant overlap do not change the integrated EMF budget;
- energy bookkeeping is Galilean invariant for a uniform boost, preserves positive internal energy, and follows the documented host-cell thermalization convention;
- active particles cannot be silently skipped by AMR placement;
- the feature is disabled by default and existing focused particle/SN tests still pass;
- `particles.md` states the model parameters, operator split, energy convention, and current AMR limits.
