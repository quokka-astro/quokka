# Local patch record

- Converted one-based Microphysics arrays to zero-based fixed-size AMReX arrays.
- Replaced generated network globals and `burn_t` callbacks with a Quokka-owned network concept.
- Made error-control participation an explicit per-variable network property; photoionization flux attenuation is integrated but passive in convergence and rejection.
- Returned structured diagnostics instead of mutating a global/concrete burn-state status contract.
- Kept partial-pivot dense LU behind a network-independent interface.
- Limited the initial extraction to ROS2S and RODAS5P, the tableaus used by the migrated photoionization and primordial-chemistry problems.
