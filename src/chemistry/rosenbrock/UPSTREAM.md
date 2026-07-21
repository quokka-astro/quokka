# Vendored Rosenbrock provenance

The Rosenbrock stage construction, ROS2S and RODAS5P coefficients, adaptive H211b controller, validity protocol, and dense linear solve in this directory were extracted from AMReX-Astrophysics Microphysics commit `b5b650048ba5da7f7caa00d6c41e71f04da905e4` on 2026-07-20.

Upstream files:

- `integration/Rosenbrock/rosenbrock_integrator.H`
- `integration/Rosenbrock/rosenbrock_tableau.H`
- `integration/Rosenbrock/rosenbrock_type.H`
- `integration/integrator_setup_strang.H`
- `integration/integrator_type_strang.H`
- `util/linpack.H`

## Local adaptation

The implementation uses zero-based `amrex::GpuArray` storage and a Quokka-owned network concept. Tolerances, validity, endpoint cleanup, and error-control participation are supplied through explicit value objects. In particular, passive integrated variables are omitted from the error norm without preprocessor knowledge in the solver. Microphysics `burn_t`, generated headers, and global runtime-parameter namespaces are not used.

The extracted tableaus are ROS2S, used by the migrated photoionization operators, and RODAS5P, used by the primordial-chemistry operators and retained as the default. Unsupported tableau values fail explicitly.

## Updating

1. Check out the new Microphysics revision separately; do not replace this directory wholesale.
2. Diff the upstream files listed above against the pinned revision.
3. Port numerical changes into the Quokka network-independent interfaces and update this document with the new revision and local differences.
4. Run the Rosenbrock unit test and the one-zone photoionization differential tests on CPU and an available GPU backend.
