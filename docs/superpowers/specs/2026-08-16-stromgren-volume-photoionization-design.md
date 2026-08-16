# Strömgren-Volume Photoionization Feedback — Design

**Date:** 2026-08-16
**Status:** Draft, awaiting review
**Scope:** New subgrid feedback module in `src/particles/`, one new 3D test problem.

## 1. Motivation

Quokka already solves photoionization accurately with the M1 two-moment solver plus the hydrogen photochemistry network (`src/radiation/photochemistry.hpp`, documented in `docs/markdown/photoionization.md`). That path is correct but expensive, and it only works when the H II region is resolved by several cells. In galaxy-scale and giant-molecular-cloud-scale runs the H II region around a young star is often smaller than one cell, so the M1 solve costs a great deal and still cannot represent the feedback.

This module provides the cheap alternative used by most galaxy-formation codes: no radiative transfer solve at all. Given the ionizing photon rate of a star particle, we find the volume around it whose recombinations exactly consume that photon budget, mark the gas inside as ionized, and hold it at a fixed temperature. The overpressure that results drives the H II region expansion and the associated momentum feedback.

## 2. Literature Review

### 2.1 The Strömgren volume technique

The method originates with Kessel-Deynet & Burkert (2000) and Dale et al. (2007b) in the SPH context. The core idea is that the ionization balance is evaluated geometrically rather than by transporting photons. For a source emitting $Q$ ionizing photons per second into gas of hydrogen number density $n_H$, ionization balance in a fully ionized, uniform medium gives the classical Strömgren radius

$$R_{\rm St} = \left(\frac{3 Q}{4\pi\,\alpha_B\,n_H^2}\right)^{1/3},$$

where $\alpha_B$ is the case-B recombination coefficient. Dale et al. generalize this by integrating the recombination rate along the line of sight from the source to each particle, treating the density structure along that ray as if it were the radial profile of a spherically symmetric distribution. Each particle is then assigned a wholly ionized or wholly neutral state each timestep. Dale et al. (2007b) extended the basic scheme to handle the dynamical case, where neutral gas can flow into an existing H II region and ionized gas can flow out or be cut off from its photon supply.

The whole approach rests on the **on-the-spot approximation**: diffuse ionizing photons produced by recombinations directly to the ground state are assumed to be reabsorbed within the same resolution element, which is why $\alpha_B$ rather than $\alpha_A$ appears. This is the same approximation already documented for Quokka's M1 module.

Dale notes that assigning a binary ionized/neutral state per particle is simpler and more robust than integrating the time-dependent ionization equations, which is the property we want here.

### 2.2 FIRE-2

Hopkins et al. (2018) use this as the local part of their LEBRON radiation scheme. Around each star particle they perform an outward gas-neighbour search, consuming the ionizing photon budget cell by cell against the local recombination rate until the budget is exhausted. Gas so flagged is held near $10^4$ K. In FIRE-2 the photons left over after the local search are handed to a long-range tree-based transfer step; in FIRE-1 the search simply terminated at the local domain boundary and the remainder was discarded. The vast majority of ionizing photons are absorbed locally, so the leftover fraction is small.

FIRE-2 splits the stellar spectrum into five bands (ionizing, far-UV, near-UV, optical/near-IR, mid/far-IR); only the ionizing band drives this module.

### 2.3 Full-transfer codes as the accuracy reference

STARFORGE (Grudić et al.), Arepo-RT (Kannan et al. 2019), and Ramses-RT solve the transfer directly rather than approximating it. They serve as the accuracy reference rather than as a model to copy. Arepo-RT validates against the classical static Strömgren sphere with a monochromatic 13.6 eV source in a uniform medium, which is the same check we adopt below.

### 2.4 Validation benchmarks

The standard dynamical benchmark is the StarBench D-type expansion comparison (Bisbas et al. 2015), in which twelve codes were run on the same problem. Two analytic solutions are commonly quoted. The Spitzer (1978) solution is

$$R_{\rm Sp}(t) = R_{\rm St}\left(1 + \frac{7 c_i t}{4 R_{\rm St}}\right)^{4/7},$$

and the Hosokawa & Inutsuka (2006) solution is

$$R_{\rm HI}(t) = R_{\rm St}\left(1 + \frac{7}{4}\sqrt{\frac{4}{3}}\,\frac{c_i t}{R_{\rm St}}\right)^{4/7},$$

with $c_i$ the sound speed in the ionized gas. StarBench found that Spitzer underestimates the numerical result by roughly 8 per cent because it neglects the inertia of the entrained shell, while Hosokawa–Inutsuka agrees much better. Bisbas et al. also give a semi-empirical fit accurate to below 2 per cent. If we test against an analytic expansion law, Hosokawa–Inutsuka is the one to use.

## 3. Design Decisions Taken

The following were settled with the user before design:

| Question | Decision |
|---|---|
| Role of the module | Standalone cheap subgrid feedback. No radiation solve involved. |
| Photon source | Existing Quokka `Star` particles. |
| Geometry of the ionized region | Distance-ordered consumption of the photon budget, i.e. a single Strömgren radius per source. |
| Effect on the gas | Set ionized gas to a fixed temperature $T_{\rm HII}$, and record the ionized fraction $x_{\rm ion}$ as an output field. |
| Anisotropy | Not modelled in v1. Spherical only, documented as a limitation. |

Cooling suppression and radiation-pressure momentum were explicitly excluded. Cooling suppression turns out to be unnecessary as a separate feature, because the temperature is reimposed every step (Section 4.5), which acts as a floor in ionized cells.

## 4. Algorithm

### 4.1 The key simplification

A literal FIRE-2 implementation gathers the cells near a star, sorts them by distance, and walks outward. On an AMReX grid that requires collecting a cell list across boxes, MPI ranks and refinement levels, which is awkward and expensive.

We avoid it by observing that **the distance-ordered walk always terminates on a ball**. Because cells are consumed in order of increasing distance, the consumed set is a prefix in distance, and a distance-prefix is by definition a sphere. Therefore the walk is equivalent to finding the single radius $R_{\rm St}$ at which

$$\int_{r < R_{\rm St}} \alpha_B\, n_e n_{H^+}\, dV = Q .$$

This equivalence is exact for an arbitrary, non-uniform density field. Non-uniform density changes the value of $R_{\rm St}$ — it is then no longer the analytic expression of Section 2.1, but whatever radius makes the actual summed recombination rate equal $Q$ — but it does not change the fact that the ionized region is a sphere.

What non-uniform density does affect is the physical fidelity of the approximation itself. A real H II region in a clumpy medium is not spherical: it breaks out along low-density directions and is blocked by dense clumps. Neither the FIRE-2 walk nor this reformulation captures that. This is a property of the chosen algorithm, not of the reformulation, and it is recorded as a limitation in Section 7.

### 4.2 Radial binning

We evaluate the integral by binning, which parallelizes cleanly and needs no sorting.

For each source, define bins of uniform width $\Delta r = \Delta x / 2$ out to a cap $R_{\rm max}$, giving $N_{\rm bin} = 2 R_{\rm max}/\Delta x$ bins. Each rank loops over its local cells within $R_{\rm max}$ of the source and accumulates into bin $k$ the local recombination rate

$$C_k = \sum_{{\rm cells\ in\ bin\ }k} \alpha_B(T_{\rm HII})\, n_H^2\, V_{\rm cell}.$$

We evaluate the recombination rate assuming the gas inside the region is fully ionized, so $n_e = n_{H^+} = n_H$. This is the standard Strömgren closure and it removes what would otherwise be a circular dependence of $x_{\rm ion}$ on itself. Because $T_{\rm HII}$ is fixed, $\alpha_B$ is a constant and can be hoisted out of the loop.

A single `ParallelDescriptor::ReduceRealSum` over the $N_{\rm bin}$ array (a few hundred bytes) completes the reduction. Every rank then forms the cumulative sum locally and finds the bin $k^*$ where the running total first exceeds $Q$. All cells in bins $k < k^*$ get $x_{\rm ion} = 1$. Cells in bin $k^*$ get the fractional value

$$x_{\rm ion} = \frac{Q - \sum_{k<k^*} C_k}{C_{k^*}},$$

applied uniformly across that shell. Cells beyond $k^*$ are untouched.

Spreading the leftover budget uniformly over the boundary shell is deliberate. A strict distance sort would have to break ties among the many cells that sit at the same radius on a Cartesian grid, and an arbitrary tie-break produces a ragged, grid-aligned boundary. The uniform shell fraction is smoother and better behaved, and it converges to the same answer as $\Delta r \to 0$.

### 4.3 Unresolved H II regions

No special case is needed. If $R_{\rm St}$ falls below $\Delta x$, the termination bin is the innermost one and the host cell simply receives a fractional $x_{\rm ion}$ equal to the fraction of its own recombination rate that the photon budget can pay for. This is exactly the subgrid behaviour the module exists to provide.

### 4.4 Multiple and overlapping sources

Sources are processed sequentially in a deterministic order: descending ionizing photon rate, with the particle id as tie-breaker. A shared ionized mask is carried across sources within a step.

When a source's walk reaches a cell already ionized by an earlier source, that cell is skipped: it is not charged against the current source's budget, and its $x_{\rm ion}$ is not reduced. This is physically right — the recombinations in that cell are already being paid for by the earlier star, so the later star's photons pass through and travel further. It also prevents the same photons being spent twice.

This costs one MPI reduction per source per step. For the test problem and for modest star counts that is negligible. Batching all sources into one reduction (an $N_{\rm src} \times N_{\rm bin}$ array) is a straightforward optimization if profiling ever shows it matters, but is not implemented in v1.

### 4.5 Applying the feedback to the gas

For each cell with $x_{\rm ion} > 0$, we raise the internal energy toward the value corresponding to $T_{\rm HII}$, weighted by the ionized fraction, and never cool the gas:

$$e_{\rm int}^{\rm new} = e_{\rm int}^{\rm old} + x_{\rm ion}\,\max\!\left(0,\; e_{\rm int}(T_{\rm HII}) - e_{\rm int}^{\rm old}\right).$$

Density and momentum are unchanged, so this is a pure heating operation; the total energy is updated consistently. Taking the `max` means gas already hotter than $T_{\rm HII}$ — shocked or supernova-heated gas — is left alone rather than being artificially cooled.

Because this is reapplied every step, it acts as a temperature floor inside the H II region for as long as the star keeps it ionized. That is why no separate cooling-suppression flag is needed.

The ionized fraction is written into a passive scalar component so that it appears in plotfiles. It is fully recomputed each step from the current density field and the current photon budget, so it is never advected as a conserved quantity; the scalar is a diagnostic output, not a state variable.

### 4.6 Where it runs in the timestep

The module is called from `AMRSimulation::particleMeshInteraction` in `src/simulation.hpp`, immediately after the supernova deposition, and operates on `finest_level`. This follows the existing convention in that function, which already assumes star particles live on the finest level.

The consequence is that the ionized region is only applied to cells covered by the finest level. If $R_{\rm St}$ extends past the finest-level grids around the star, the outer part of the H II region is missed. Rather than fail silently, the module emits a runtime warning when the Strömgren radius exceeds the extent of finest-level coverage around a source, or when the budget is not exhausted within $R_{\rm max}$.

### 4.7 Obtaining the ionizing photon rate

Two paths, because the test problem needs an exact rate and production runs need one tied to stellar properties:

1. **Fixed rate** — input parameter `stromgren.Q_ion`, applied to every star particle. Used by the test problem, where an exact $Q$ is required to compare with the analytic radius.
2. **Derived from luminosity** — $Q = f_{\rm ion} L_0 / (h\bar\nu)$, where $L_0$ is the particle's first luminosity slot (already populated by the stellar-evolution model), $f_{\rm ion}$ is `stromgren.ionizing_fraction`, and $h\bar\nu$ is `stromgren.mean_photon_energy` (default 18 eV).

`Physics_Traits::nGroups` defaults to 1, so a `Star` particle carries a luminosity slot even when radiation is disabled. No new particle components are needed for either path.

## 5. Code Layout

| File | Contents |
|---|---|
| `src/particles/particle_photoionization.hpp` | New. The whole module: parameter struct, the binning reduction, the cumulative-sum solve for $R_{\rm St}$, and the gas update. Header-only, matching the style of the other `src/particles/` modules. |
| `src/simulation.hpp` | One call added inside `particleMeshInteraction`, guarded so it is a no-op when the module is off. |
| `src/problems/StromgrenVolumeFeedback/` | New 3D test problem: `testStromgrenVolumeFeedback.cpp`, `CMakeLists.txt`. |
| `inputs/StromgrenVolumeFeedback.toml` | Test problem input. |
| `docs/markdown/photoionization.md` | New section describing the subgrid module and when to use it instead of M1. |

Input parameters, all under the `stromgren.` prefix: `enabled`, `Q_ion`, `ionizing_fraction`, `mean_photon_energy`, `T_HII` (default $10^4$ K), `R_max` (default 32 cells), `x_ion_scalar_index`.

The module is a single header of a few hundred lines with one public entry point taking the state MultiFab, the particle container, the geometry and the level. It depends only on the particle container and the EOS, and it can be understood and tested without reference to the radiation module.

## 6. Testing

**Primary test — `StromgrenVolumeFeedback` (3D, in CI).** A uniform medium at $n_H = 10^3\ \mathrm{cm}^{-3}$ with a single star particle at the domain centre emitting $Q = 10^{49}\ \mathrm{s}^{-1}$. The analytic radius is $R_{\rm St} \approx 0.68$ pc. On a 4 pc box at $64^3$ this is about 11 cells, so the region is well resolved and the run is cheap.

Hydro stays enabled, because the module runs inside `particleMeshInteraction` and operates on the hydro state; running with `is_hydro_enabled = false` is an untested configuration for the star-particle path and should not be introduced here. Instead the stop time is set to a small fraction of the sound-crossing time of $R_{\rm St}$, so the gas has not moved appreciably and the density is still uniform when the check is applied. The test then integrates $x_{\rm ion}$ over the grid to get the ionized volume and compares the equivalent radius with $R_{\rm St}$, requiring agreement to better than one cell width. Because the density is still uniform at that point, the comparison is exact up to the discretization of the boundary shell, so a tight tolerance is justified.

For quick local iteration the same problem runs at $32^3$ with a correspondingly shorter stop time; the CI configuration uses $64^3$.

**Secondary check, same binary.** With the source placed off-centre and a deliberately imposed density gradient, verify that the total recombination rate inside the flagged region equals $Q$ to round-off. This tests the non-uniform-density path — the part where the analytic formula no longer applies but photon conservation still must hold.

**Regression check.** `ParticleSinkFormation` or `ParticleSF` (3D), which exercises the star-particle path this module hooks into but does not itself change, to confirm the `particleMeshInteraction` edit breaks nothing.

**Not in CI: D-type expansion.** The full StarBench comparison against Hosokawa–Inutsuka needs hydro on and enough resolution and runtime to be a poor fit for the test suite. Worth running once by hand as a validation exercise, with the result recorded in the PR, but not added as an automated test.

## 7. Known Limitations

These are deliberate and should be stated in the user-facing documentation.

1. **The ionized region is always spherical.** Champagne flows, breakout along low-density channels, and shadowing behind dense clumps are not represented. This is inherent to the Strömgren-volume approximation as chosen; the fix would be angular binning, which was considered and deferred.
2. **Finest-level only.** H II regions extending beyond the finest-level grids are truncated. The module warns when this happens.
3. **Photons are not conserved globally.** A budget not exhausted within $R_{\rm max}$ is discarded, as in FIRE-1. There is no long-range transfer step to catch the remainder.
4. **The mean molecular weight does not track ionization.** Quokka's EOS uses a fixed $\mu$ unless mass scalars are evolved, so ionized gas keeps its neutral $\mu$. Since the pressure is $P = \rho k T/(\mu m_H)$, the overpressure driving the expansion is underestimated by roughly the ratio $\mu_{\rm neutral}/\mu_{\rm ionized} \approx 2$. Users should compensate by setting $T_{\rm HII}$ to an effective value of about $2\times10^4$ K rather than $10^4$ K. This is a documentation note; making $\mu$ ionization-dependent is out of scope.
5. **No radiation pressure.** Only thermal feedback, by explicit choice.

## 8. Open Question for Review

Section 4.7 assumes the luminosity-derived path is worth having in v1. If star particles in the intended production runs will always set $Q$ some other way, that path could be dropped and the module reduced to the fixed-rate parameter plus a user-supplied hook. Please confirm.

## 9. References

- Bisbas et al. (2015), *StarBench: the D-type expansion of an H II region*, MNRAS 453, 1324. [arXiv:1507.05621](https://arxiv.org/abs/1507.05621)
- Dale, Ercolano & Clarke (2007b), and Dale et al. (2005), *Photoionizing feedback in star cluster formation*, MNRAS 358, 291. [arXiv:astro-ph/0501160](https://arxiv.org/pdf/astro-ph/0501160)
- Hopkins et al. (2018), *FIRE-2 simulations: physics versus numerics in galaxy formation*, MNRAS 480, 800, Appendix E. [arXiv:1702.06148](https://arxiv.org/pdf/1702.06148)
- Hosokawa & Inutsuka (2006).
- Kannan et al. (2019), *Arepo-RT: radiation hydrodynamics on a moving mesh*. [arXiv:1804.01987](https://arxiv.org/pdf/1804.01987)
- Kessel-Deynet & Burkert (2000).
- Grudić et al. (2021), *STARFORGE*. [arXiv:2010.11254](https://arxiv.org/pdf/2010.11254)
- Spitzer (1978), *Physical Processes in the Interstellar Medium*.
- Dale, *The modelling of feedback in star formation simulations*. [NED review](https://ned.ipac.caltech.edu/level5/Sept15/Dale/Dale2.html)
