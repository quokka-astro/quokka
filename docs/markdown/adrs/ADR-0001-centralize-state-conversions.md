# ADR-0001: Centralize Conserved-To-Primitive And Primitive-To-Conserved State Conversions
Date: 2026-03-28 • Status: Proposed

## Context

Quokka currently assumes a mostly uniform interpretation of the stored hydro state:

- `rho`
- `rho v`
- `rho E`
- `rho e`

That assumption is embedded in multiple places across the codebase. Some modules already use shared helpers such as `ConservedToPrimitive`, but other paths directly interpret conserved components by

- dividing momentum by density to recover velocity,
- subtracting kinetic energy from total energy to recover internal energy,
- computing temperature or pressure from locally reconstructed quantities,
- applying source terms directly to conserved components.

This is acceptable while the stored state is always the physical conserved state, but it becomes unsafe if Quokka adopts a different storage convention such as cosmological comoving variables:

- `rho`
- `a rho v`
- `a^2 rho E`
- `a^2 rho e`

In that case, any module that directly interprets conserved components as physical variables will silently become inconsistent.

This issue affects multiple subsystems:

- hydro reconstruction and flux evaluation,
- dual-energy bookkeeping,
- floors and validity checks,
- MHD state interpretation and any Boris-corrected momentum-to-velocity transforms,
- source modules such as cooling and chemistry,
- gravity coupling,
- diagnostics and derived variables,
- future cosmology support.

This ADR therefore addresses an architectural prerequisite for cosmological hydro and for Boris-corrected MHD: Quokka must centralize state conversion logic and stop allowing widespread ad hoc interpretation of conserved state.

The Boris correction strengthens the case for this change. In a Boris-corrected MHD formulation, the conserved momentum advanced by the hyperbolic solver is no longer identical to the physical fluid momentum `rho v`. Instead, it is modified by the electromagnetic contribution associated with the Poynting flux, which can be viewed as an effective inertia rescaling by

```text
1 + v_A^2 / c^2
```

where `v_A` is the Alfvén speed and `c` is the reduced speed entering the Boris correction. In that setting, directly dividing the stored momentum by density is no longer a valid way to recover the physical velocity. Reconstruction, wave-speed estimates, source terms, and diagnostics all need one authoritative mapping between stored conserved variables and physical primitive variables. Without a centralized conversion layer, that mapping would be vulnerable to the same class of silent inconsistency as a comoving cosmological variable formulation [@Boris_1970].

For example, if a Boris-corrected formulation stores

- `rho`,
- `B`,
- modified momentum `M = rho v + S/c^2`,
- total energy `E`,

with ideal-MHD Poynting flux

```text
S = B^2 v - (v·B) B,
```

then the modified momentum is

```text
M = (rho + B^2/c^2) v - (v·B) B / c^2.
```

Decomposing into components parallel and perpendicular to `B`,

```text
M_parallel = rho v_parallel
M_perp = (rho + B^2/c^2) v_perp,
```

so the corresponding conserved-to-primitive inversion is

```text
v_parallel = M_parallel / rho
v_perp = M_perp / (rho + B^2/c^2)
       = M_perp / [rho (1 + v_A^2/c^2)].
```

In vector form, with `b = B / |B|`,

```text
v = (M·b)/rho * b + [M - (M·b) b] / (rho + B^2/c^2).
```

This is exactly the kind of nontrivial `cons2prim` mapping that should live in one canonical helper layer rather than being rederived piecemeal in reconstruction, Riemann solves, diagnostics, or source updates.

Touches:

- problem-file interface: no user hook changes are required immediately, but problem modules that directly inspect conserved state may need migration guidance,
- `ParmParse` options: no immediate changes required,
- output format: no changes required.

## Options

- Option A: Keep the current pattern and audit call sites only when cosmology is added.
- Option B: Centralize all conserved-to-primitive and primitive-to-conserved mappings behind canonical helper functions and migrate modules to use them.
- Option C: Add a cosmology-specific conversion layer only for the new code path and leave the existing physical-state path structurally unchanged.

## Decision

Choose Option B.

Quokka should define a canonical conversion layer for hydro state interpretation and make it the only supported path for translating between stored conservative variables and physical primitive variables.

The core requirements are:

- `ConservedToPrimitive` becomes the canonical path from stored hydro state to physical primitive variables,
- the inverse mapping to the stored conserved state is made equally explicit and reusable,
- shared helper functions are added for common physical queries such as velocity, kinetic energy, gas pressure, total/internal energy, and temperature,
- modules are migrated away from directly interpreting conserved components where a canonical helper exists,
- new features, especially cosmology support and Boris-corrected MHD, must build on this conversion layer rather than introducing local one-off interpretations.

This decision is motivated by correctness and maintainability. A conversion layer is required to safely support alternate stored-state conventions, and it also reduces the risk of inconsistent physical-state reconstruction even in the current non-cosmological path.

## Consequences

Positive consequences:

- Quokka can support alternate stored-state conventions, including comoving cosmological variables, without relying on fragile code-wide assumptions.
- Quokka can support Boris-corrected MHD without scattering momentum-to-velocity conversion logic across hydro, MHD, source, and diagnostic code.
- Variable semantics become explicit at module boundaries.
- Derived-variable calculations, source terms, and diagnostics can share one consistent interpretation of the hydro state.
- Future audits become easier because the number of sanctioned conversion paths is small and well documented.

Negative consequences:

- This is a nontrivial refactor touching multiple modules.
- Some existing code paths that currently perform local algebra directly on conserved components will need to be rewritten to use helpers.
- During migration, temporary duplication may exist while old call sites are replaced.
- There is a short-term maintenance cost to define and document helper APIs carefully enough that they are usable on CPU and GPU paths.

Performance impact:

- The intended design should preserve performance by keeping helpers inline and GPU-safe.
- Some call sites may become slightly more abstract, so hot paths must be checked to ensure no unintended overhead is introduced.

Portability impact:

- Helpers must remain valid for CPU, CUDA, and HIP builds.
- Any conversion API used inside device kernels must obey Quokka's existing GPU-safety constraints.

Migration impact:

- No restart-format or plotfile-format changes are required by this ADR alone.
- Problem modules and source modules may need updates if they currently infer physical state directly from conserved components.

## Rollout & Testing

### Rollout plan

1. Identify the canonical conversion API surface.
2. Implement or formalize the inverse `prim2cons` mapping alongside `ConservedToPrimitive`.
3. Add small shared helpers for common physical queries used outside the reconstruction path.
4. Audit hydro, dual-energy, floors, diagnostics, gravity, cooling, and chemistry for direct conserved-state interpretation.
5. Audit MHD paths for direct momentum-to-velocity or energy interpretation that would conflict with a Boris-corrected mapping.
6. Replace direct call sites incrementally with canonical helpers.
7. Add assertions and comments documenting when quantities are physical versus stored-state quantities.
8. Make cosmology support and any Boris-corrected MHD path depend on this layer rather than bypassing it.

### Testing plan

- Existing hydro regression tests must continue to pass unchanged.
- Existing MHD regression tests must continue to pass unchanged before any Boris-correction feature is enabled.
- Add unit or integration coverage for round-trip conversion where practical:
  - physical primitive -> stored conserved -> physical primitive
  - stored conserved -> physical primitive -> stored conserved
- Add tests covering dual-energy consistency under the helper-based conversion path.
- When cosmology is introduced, add dedicated tests ensuring the same helper layer supports both `a = 1` and expanding-background cases.
- When Boris-corrected MHD is introduced, add tests ensuring the same helper layer consistently recovers the physical velocity and pressure state used by reconstruction, Riemann solves, and diagnostics.

## Links

- Cosmology design note: [`docs/markdown/cosmology_plan.md`](/Users/benwibking/amrex_codes/quokka/docs/markdown/cosmology_plan.md)
- ADR guidance: [`docs/markdown/adrs.md`](/Users/benwibking/amrex_codes/quokka/docs/markdown/adrs.md)
- Hydro integrator overview: [`docs/markdown/hydro_integrator.md`](/Users/benwibking/amrex_codes/quokka/docs/markdown/hydro_integrator.md)
- Reference bibliography entry: `Boris_1970` in [`docs/markdown/references.bib`](/Users/benwibking/amrex_codes/quokka/docs/markdown/references.bib)
