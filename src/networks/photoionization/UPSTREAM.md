# Upstream provenance

`PhotoionizationNetwork.hpp` was adapted from `networks/photoionization/actual_rhs.H` and `EOS/photoionization/actual_eos.H` in AMReX-Astrophysics Microphysics commit `b5b650048ba5da7f7caa00d6c41e71f04da905e4` on 2026-07-20.

The algebra is exposed through the Quokka chemistry-network contract, runtime parameters are stored in a value object, radiation-flux attenuation is explicitly marked as integrated but passive for error control, and the network-owned multigamma thermodynamic operations replace Microphysics EOS state inside chemistry updates. Species masses and the Boltzmann constant are frozen to the values used by that upstream revision.

To update, compare the upstream RHS, Jacobian, EOS constants, and parameter defaults against `PhotoionizationNetwork.hpp`; port deliberate changes into the Quokka adapter; update the pinned revision here; and run `RosenbrockChemistry`, both `OneZonePhotoionization` cases, and a representative multidimensional ionization-front test on CPU and an available GPU backend.
