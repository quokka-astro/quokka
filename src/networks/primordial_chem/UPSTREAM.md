# Primordial chemistry network provenance

`GeneratedRhs.hpp` was imported from `networks/primordial_chem/actual_rhs.H` in AMReX-Astrophysics Microphysics commit `b5b650048ba5da7f7caa00d6c41e71f04da905e4` on 2026-07-20.

The generated algebra is unchanged. The local import places it in a namespace, substitutes a narrow Quokka-owned RHS state, and accepts redshift as an explicit immutable parameter rather than reading a generated global. `PrimordialChemNetwork.hpp` owns species metadata, mass and heat-capacity data, thermodynamic conversion, charge closure, validity, and the adapter to the shared Rosenbrock integrator.

To update, regenerate or obtain the new upstream `actual_rhs.H`, compare it with `GeneratedRhs.hpp` while ignoring the wrapper substitutions described above, reapply those substitutions, update the pinned revision here, and run `PrimordialChem` plus `RosenbrockChemistry` on CPU and an available GPU backend.
