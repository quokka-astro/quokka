# SNIa, oldAGB, and NSM development plan

## Status and scope

This is a development roadmap, not a description of implemented physics.
SNII, WR, and AGB feedback were merged in
[PR #1915](https://github.com/quokka-astro/quokka/pull/1915) and form the baseline.
This follow-up tracks SNIa, oldAGB, and neutron-star merger (NSM) feedback.
The initial PR commit contained documentation only. The working tree now has a
selective SNIa prototype port, which has not been built or validated. SNIa remains
disabled by default; oldAGB and NSM are not implemented.

An unmerged local SNIa prototype exists on an older baseline. It contains
event-based hydrodynamic and chemical deposition helpers, a SNIa yield-table
extension, a single-event test, and a TallBox event-sampling experiment.
It also changes stellar mass sampling and particle component layouts.
Only event deposition, SNIa table loading, channel storage, and a compact test
have been adapted to the merged baseline. IMF changes, TallBox sampling,
visualization scripts, and obsolete CTest fixtures were not ported.
No oldAGB or NSM implementation has been identified in that prototype.

### Current SNIa interface and limitations

`particles/particle_snia_feedback.hpp` provides hydrodynamic event deposition
and SNIa isotope deposition using the existing SN kernels. Both are collective
operations: every rank must call them with the same ordered event list, including
positions and velocities. Count agreement is checked, but matching coordinates
remain the caller's responsibility. Events must lie inside the physical domain;
the valid cell containing an event determines which rank deposits it. State ghost
cells must be filled before hydrodynamic deposition. The interface is currently
three-dimensional and does not provide AMR scheduling or an event-rate model.

Call hydrodynamic and chemical deposition once each for a physical event; do not
call them again during particle feedback or subcycling. The SNIa example uses a
single unrefined periodic level and a replicated deterministic event. It includes
assertions for empty event lists, event count, ejecta mass, isotope masses, and
zero contribution to other channels, but these assertions have not been run.

SNIa data are distributed as a separate CSV rather than replacing the existing
yield archive. Particle history storage grows to five blocks for chemistry-enabled
problems. Existing four-block checkpoints need an explicit migration strategy;
only the repository's two yield-test ASCII fixtures are adapted here.

Validation, formatting/lint checks, and numerical model review are pending. The
port is not a completed or production-ready SNIa implementation.

## Implementation sequence

### 1. Port and validate the SNIa prototype

- [ ] Separate reusable deposition code from TallBox-specific event generation.
- [ ] Review event ownership across MPI ranks, empty local event lists, collective
  operations, and deposition across grid and periodic boundaries.
- [ ] Adapt table loading and channel indexing without reverting the opt-in
  chemistry storage, shared channel counts, or configure-time table extraction.
- [ ] Audit yield-table provenance, units, normalization, and reproducibility.
- [ ] Add a compact SNIa test using the current problem naming and CTest helpers,
  with numerical assertions rather than plot output alone.
- [ ] Decide whether the prototype's IMF changes are required; do not make them
  an implicit dependency of SNIa event deposition.

### 2. Specify and implement oldAGB

- [ ] Define what the working label `oldAGB` represents, including the source
  population, age/mass range, and whether it is resolved or population-averaged.
- [ ] Specify its relationship to the existing AGB channel and prevent double
  counting of ejecta from the same stellar population.
- [ ] Select and document yield data, time dependence, normalization, and units.
- [ ] Implement mass and isotope return with independently checked integrated
  budgets and timestep-boundary tests.

### 3. Specify and implement NSM

- [ ] Define the event-rate/delay model, source population, and spatial sampling.
- [ ] Select and document ejecta yields, tracked species, mass, and energy inputs.
- [ ] Implement reproducible event generation and conservative deposition using
  shared infrastructure where appropriate.
- [ ] Validate individual events and integrated event/yield budgets.

## Shared design decisions

- Define channel identifiers and diagnostic fields centrally; do not introduce
  unrelated literal channel counts or silently change existing field meanings.
- Establish disabled-channel behavior and compatibility with existing inputs.
- Decide particle/checkpoint layout compatibility before extending storage.
- Specify restart handling for pending events and random-number generator state.
- Keep GPU captures device-safe and MPI collectives consistent across ranks.
- Define the timestep and AMR level at which each source is applied, avoiding
  duplicate feedback under subcycling or particle evolution transitions.

## Validation required before leaving draft

- [ ] Existing SNII and WR/AGB yield tests remain passing with new channels off.
- [ ] New tests check total and per-channel isotope budgets, nonzero and zero
  yields, and mass/energy/momentum conservation as applicable.
- [ ] Event tests cover zero events, multiple events, timestep boundaries, and
  restart reproducibility.
- [ ] Single-rank and multi-rank runs agree within documented tolerances.
- [ ] CPU and available GPU builds/runs are recorded separately; compilation is
  not reported as runtime validation.
- [ ] FPE coverage follows `ENABLE_TESTS_FPE`, including existing platform policy.
- [ ] clang-tidy and repository pre-commit checks pass.

Model choices and numerical parameters remain open until explicitly documented
and reviewed. The existing SNII/WR/AGB implementation is not reopened by this plan.
